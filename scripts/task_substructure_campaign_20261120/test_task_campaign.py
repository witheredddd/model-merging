"""Linux integration smoke test: fresh start, supplement, packaging and resume."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path

class CampaignTest(unittest.TestCase):
    def test_fresh_campaign_and_resume_do_not_repeat_completed_jobs(self):
        base=Path(os.environ.get('CAMPAIGN_TEST_ROOT','/media/shangxueyuan/data1'))
        if not base.exists():self.skipTest('Run on the A100 Linux host')
        script=Path(__file__).with_name('task_substructure_campaign.py')
        with tempfile.TemporaryDirectory(prefix='task_campaign_test_',dir=base) as tmp:
            p=Path(tmp)
            for name in ('pilot','small','large','helpers','trace'):(p/name).mkdir()
            (p/'trace/trace.json').write_text(json.dumps({'rows':[{'step':0},{'step':25}]}))
            worker='''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a,_=p.parse_known_args()
(a.output/'bases').mkdir(exist_ok=True)
for anchor in (0,25):(a.output/f'bases/a{anchor}.json').write_text('{}')
(a.output/'status.json').write_text('{"stage":"COMPLETE"}')
with (a.output/'executions.txt').open('a') as f:f.write('primary\\n')
'''
            supplement='''import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a,_=p.parse_known_args()
(a.output/'SUPPLEMENT_COMPLETE.json').write_text('{}')
(a.output/'status.json').write_text('{"stage":"COMPLETE"}')
with (a.output/'executions.txt').open('a') as f:f.write('supplement\\n')
'''
            report='''import sys
from pathlib import Path
p=Path(sys.argv[1])/'analysis'/('anchor_'+sys.argv[3]);p.mkdir(parents=True,exist_ok=True)
(p/'SUMMARY.json').write_text('{"complete":true}')
'''
            (p/'small/worker.py').write_text(worker)
            (p/'helpers/task_substructure_supplement.py').write_text(supplement)
            (p/'helpers/task_substructure_report.py').write_text(report)
            cmd=[sys.executable,str(script),'--root',str(p/'campaign'),'--pilot-output',str(p/'pilot'),
                '--pilot-unit','nonexistent-test-substructure.service','--pilot-scripts',str(p/'small'),
                '--large-scripts',str(p/'large'),'--helpers',str(p/'helpers'),'--trace',str(p/'trace'),'--skip-large']
            for name in ('inventory','probes','coordinates','flow-source'):cmd+=['--'+name,str(p/name)]
            first=subprocess.run(cmd,capture_output=True,text=True,timeout=100)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            before=(p/'pilot/executions.txt').read_text()
            second=subprocess.run(cmd,capture_output=True,text=True,timeout=100)
            self.assertEqual(second.returncode,0,second.stdout+second.stderr)
            self.assertEqual(before,(p/'pilot/executions.txt').read_text())
            self.assertEqual(json.loads((p/'campaign/CAMPAIGN_STATUS.json').read_text())['stage'],'COMPLETE')
            self.assertTrue((p/'pilot/RESULT_PACKAGE/REPORTS.tar.gz').exists())

if __name__=='__main__':unittest.main()
