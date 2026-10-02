"""Real inference smoke study; no ground-truth 3D accuracy claim.

python scripts/validate_depth.py --experiments /path/gargoyles.zip /path/cats.zip \
  --output /path/results --max-size 512 --detail 6
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from modern_cliche.pipeline import prepare, load_experiment, subset
from modern_cliche.depth import infer_cohort, depth_preview
from modern_cliche.analysis import analyse
from modern_cliche.surface import surface_mesh
from modern_cliche.export import experiment_zip, dumps, mesh_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-size', type=int, default=512)
    parser.add_argument('--detail', type=int, default=6)
    parser.add_argument('--model', default='moge-small')
    parser.add_argument('--exclude-map', type=Path, help='JSON mapping dataset folder names to audited excluded IDs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, examples = [], []
    exclusions = json.loads(args.exclude_map.read_text()) if args.exclude_map else {}
    for path in args.experiments:
        samples, overrides, manifest = load_experiment(path.read_bytes())
        prepared = prepare(samples, size=manifest['preparation']['size'], overrides=overrides)
        flags = {entry['id']: entry.get('quality', {}).get('needs_review', False) for entry in manifest['samples']}
        reject = exclusions.get(path.parent.name, [])
        keep = [i for i, sample in enumerate(prepared['samples']) if not flags.get(sample.id, False) and sample.id not in reject]
        excluded = [dict(id=sample.id, metadata=sample.metadata, reason='original-mask-quality-flag-or-audited-exclusion')
                    for i, sample in enumerate(prepared['samples']) if i not in keep]
        prepared = subset(prepared, keep)
        prepared['excluded_samples'] = excluded
        print('DATASET',path.parent.name, len(prepared['samples']),flush=True)
        def progress(value):print('depth',round(value * len(keep)), '/', len(keep),flush=True)
        prepared['depth_results'] = infer_cohort(prepared,args.model,args.max_size,args.detail,progress=progress)
        analysis = analyse(prepared['masks'], depth_results=prepared['depth_results'])
        index = analysis['groups'][0]['medoid']
        result = prepared['depth_results'][index]
        mesh, metrics = surface_mesh(result)
        shell, shell_metrics = surface_mesh(result,thickness_mm=2.)
        recipe = dict(mode='observed',specimen=index,input_indices=[index],geometry='depth')
        output = args.output / path.parent.name;output.mkdir(exist_ok=True)
        (output / 'experiment.zip').write_bytes(experiment_zip(prepared,analysis,manifest['query'],mesh,recipe,metrics,prepared['masks'][index],selected_depth=result))
        for ext in ('glb','ply','stl'):(output / f'surface.{ext}').write_bytes(mesh_bytes(mesh,ext))
        (output / 'shell.stl').write_bytes(mesh_bytes(shell,'stl'))
        depth_preview(result).save(output / 'depth.png')
        prepared['samples'][index].image.save(output / 'source.png')
        summary = dict(dataset=path.parent.name,images=len(keep),groups=len(analysis['groups']),
            excluded=excluded,
            medoid=prepared['samples'][index].id, surface=metrics,shell=shell_metrics,
            median_inference_seconds=float(np.median([r['metadata']['seconds'] for r in prepared['depth_results']])),
            valid_fractions=[float(r['features'][-1]) for r in prepared['depth_results']],
            cluster_silhouette=analysis['clusters']['silhouette'],subsample_ari=analysis['clusters']['stability'])
        summaries.append(summary);examples.append((prepared,result,mesh,index,path.parent.name))
        (args.output / 'metrics.json').write_text(dumps(summaries))
        print('DONE',path.parent.name,metrics['vertices'],metrics['faces'],flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    fig = plt.figure(figsize=(14,4*len(examples)))
    for row,(prepared,result,mesh,index,name) in enumerate(examples):
        for col in (0,1):
            ax = fig.add_subplot(len(examples),4,row*4+col+1)
            ax.imshow(prepared['samples'][index].image if col==0 else depth_preview(result));ax.axis('off')
            ax.set_title(name.replace('verified_','')+' / '+('input' if col==0 else 'predicted depth'))
        # Render the full predicted surface; no smoothing or appearance synthesis.
        small,_=surface_mesh(result,max_resolution=768)
        v,f=small.vertices,small.faces
        for col,azimuth in [(2,-75),(3,-10)]:
            ax=fig.add_subplot(len(examples),4,row*4+col+1,projection='3d')
            ax.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=f,color='#bbbbbb',linewidth=0,antialiased=False,shade=True)
            ax.set_box_aspect(np.maximum(small.extents,1));ax.view_init(elev=15,azim=azimuth)
            ax.set_title('Predicted visible surface / '+('front' if col==2 else 'side'))
            ax.set_axis_off()
    fig.tight_layout();fig.savefig(args.output/'depth_verification.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
