import sys,pathlib,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from run_toy import run

class ToyIntegration(unittest.TestCase):
    def test_online_birth_affects_real_trajectory(self):
        online=run('online',6)
        frozen=run('frozen',6)
        reference=run('reference',6)
        self.assertGreater(online['rank'],frozen['rank'])
        self.assertLess(online['final_test_ce'],online['initial_test_ce'])
        self.assertLess(online['final_test_ce'],frozen['final_test_ce'])
        self.assertLess(reference['final_test_ce'],reference['initial_test_ce'])
        for record in online['steps']:
            if record['projection_error'] is not None:
                self.assertLess(record['projection_error'],1e-10)
    def test_frozen_stops_births(self):
        frozen=run('frozen',4)
        self.assertEqual(sum(x['birth'] for x in frozen['steps']),1)
        self.assertEqual(frozen['rank'],1)

if __name__=='__main__':unittest.main()
