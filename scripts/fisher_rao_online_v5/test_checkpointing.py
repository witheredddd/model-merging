import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import torch
from trainer import OnlineTrainer,Settings
from reporting import CostReporter
from checkpointing import capture,restore,atomic_save,model_payload,archive_snapshot
from test_online_v5 import ToyOracle


def reporter_for(holder):
    return CostReporter(list(range(0,20001,32)),lambda:holder[0].state,
        lambda state:dict(CE=float(state.parameters().square().sum()),accuracy=0.,n=1),lambda state,cost:None)


class CheckpointTests(unittest.TestCase):
    def test_full_checkpoint_resume_matches_uninterrupted_geometry_and_budget(self):
        raw=ToyOracle();raw.phase=1
        settings=Settings(geometry_every=2,persistence_windows=2)
        engine=OnlineTrainer(raw,settings,budget=20000);holder=[engine];reporter=reporter_for(holder)
        reporter.bootstrap_base(engine.ledger.used);engine.ledger.on_progress=reporter.before_progress
        for _ in range(3):engine.step();reporter.flush()
        identity=dict(task='toy',source='frozen',base='same')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recovery.pt'
            atomic_save(path,capture(engine,reporter,identity,model_payload(engine.state)))
            payload=torch.load(path,map_location='cpu',weights_only=True)
        for _ in range(3):engine.step();reporter.flush()
        resumed_raw=ToyOracle();resumed_raw.phase=1
        other=[None];resumed_reporter=reporter_for(other)
        with patch('trainer.initialize',side_effect=AssertionError('Resume must not initialize')):
            resumed=restore(payload,resumed_raw,settings,identity,resumed_reporter);other[0]=resumed
        for _ in range(3):resumed.step();resumed_reporter.flush()
        torch.testing.assert_close(resumed.state.parameters(),engine.state.parameters(),atol=0,rtol=0)
        torch.testing.assert_close(resumed.state.G,engine.state.G,atol=0,rtol=0)
        torch.testing.assert_close(resumed.geometry.A,engine.geometry.A,atol=0,rtol=0)
        self.assertEqual(resumed.ledger.images,engine.ledger.images)
        self.assertEqual(resumed.ledger.operations,engine.ledger.operations)
        self.assertEqual((resumed.iteration,resumed.window,resumed.radius),(engine.iteration,engine.window,engine.radius))
        self.assertEqual(resumed.persistence.last,engine.persistence.last)
        self.assertEqual(resumed_reporter.rows,reporter.rows)

    def test_wrong_assets_or_source_are_rejected_without_initialization(self):
        engine=OnlineTrainer(ToyOracle());holder=[engine];reporter=reporter_for(holder)
        reporter.bootstrap_base(engine.ledger.used)
        payload=capture(engine,reporter,dict(source='a'),model_payload(engine.state))
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            restore(payload,ToyOracle(),engine.settings,dict(source='b'),reporter)

    def test_failed_snapshot_write_leaves_previous_checkpoint_intact(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'recovery.pt';atomic_save(path,dict(iteration=8))
            with patch('checkpointing.torch.save',side_effect=OSError('interrupted write')):
                with self.assertRaises(OSError):atomic_save(path,dict(iteration=16))
            self.assertEqual(torch.load(path,weights_only=True),dict(iteration=8))

    def test_periodic_archives_survive_latest_pointer_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            first=archive_snapshot(folder,dict(iteration=8,ledger=dict(images=dict(train=64)),value=torch.tensor([1.])))
            second=archive_snapshot(folder,dict(iteration=16,ledger=dict(images=dict(train=128)),value=torch.tensor([2.])))
            self.assertNotEqual(first,second)
            self.assertEqual(torch.load(first,weights_only=True)['value'].item(),1.)
            self.assertEqual(torch.load(Path(folder)/'recovery_state.pt',weights_only=True)['value'].item(),2.)


if __name__=='__main__':unittest.main()
