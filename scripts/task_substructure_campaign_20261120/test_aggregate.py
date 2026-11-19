import csv,tempfile,unittest
from pathlib import Path
from task_substructure_aggregate import hypervolume,main

class AggregateTest(unittest.TestCase):
    def test_hypervolume_handles_dominated_points_and_noop(self):
        self.assertAlmostEqual(hypervolume([(-2,-1),(-1,-2),(-.5,-.5)],(0,0)),3.)
        self.assertEqual(hypervolume([(1,-1)],(0,0)),0.)
        self.assertAlmostEqual(hypervolume([(0,0)],(.01,.01)),.0001)

    def test_missing_seed_not_imputed_and_resampling_label_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed,value in [(41,-.1),(42,-.2)]:
                folder=root/f'large_seed{seed}'/'analysis/anchor_0';folder.mkdir(parents=True)
                with (folder/'all_metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['anchor','family','method','demand','radius_index','sign','task_probe','delta_CE']);w.writeheader()
                    w.writerow(dict(anchor=0,family='sufficiency',method='full_gradient',demand='mean',radius_index=1,sign=1,task_probe='DTD_p3',delta_CE=value))
            main(root)
            with (root/'analysis/resampling_metrics.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['n_seed_resamplings'],'2')
            self.assertAlmostEqual(float(rows[0]['mean']),-.15);self.assertEqual(rows[0]['independent_training_trials'],'False')

if __name__=='__main__':unittest.main()
