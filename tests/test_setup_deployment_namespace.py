"""Exercise shell selectors without invoking cloud or dependency commands."""
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class SetupNamespaceTests(unittest.TestCase):
    def shell(self, values=None):
        env = {k:v for k,v in os.environ.items() if not k.startswith(('AMLAS_', 'AGENTICML_', 'TAVILY_'))}
        env.update(values or {})
        return subprocess.run(['bash','-c',
            '. scripts/lib/deploy_common.sh; configure_deployment_namespace || exit $?; '
            'printf "%s\\n" "$AMLAS_DEPLOYMENT_PREFIX" "$AMLAS_SSM_PREFIX" '
            '"$DEFAULT_AGENTCORE_RUNTIME_NAME" "$DEFAULT_TAVILY_SECRET_ID" '
            '"$AMLAS_AGENTCORE_SSM_PARAM" "$AMLAS_SESSION_BUCKET_SSM_PARAM"; '
            'deployment_stack_name LakeStack'], cwd=ROOT, env=env, text=True, capture_output=True)
    def test_legacy_default_is_preserved(self):
        x=self.shell()
        self.assertEqual(x.returncode,0,x.stderr)
        self.assertEqual(x.stdout.splitlines(),['amlas','/amlas','amlas','amlas/tavily-api-key','/amlas/agentcore-runtime-arn','/amlas/session-bucket','LakeStack'])
    def test_selected_namespace_covers_every_publication_and_stack(self):
        x=self.shell({'AMLAS_DEPLOYMENT_PREFIX':'amlas-fresh-1013'})
        self.assertEqual(x.returncode,0,x.stderr)
        self.assertEqual(x.stdout.splitlines(),['amlas-fresh-1013','/amlas-fresh-1013','amlas_fresh_1013','amlas-fresh-1013/tavily-api-key','/amlas-fresh-1013/agentcore-runtime-arn','/amlas-fresh-1013/session-bucket','amlas-fresh-1013-LakeStack'])
    def test_invalid_prefix_is_rejected_before_commands(self):
        for prefix in ('Amlas','amlas--test','amlas/other','amlas-','a'*25,'$(touch injected)'):
            with self.subTest(prefix=prefix):
                x=self.shell({'AMLAS_DEPLOYMENT_PREFIX':prefix})
                self.assertNotEqual(x.returncode,0)
                self.assertIn('deployment prefix',x.stderr.lower())
    def test_independent_ssm_namespace_is_rejected(self):
        x=self.shell({'AMLAS_DEPLOYMENT_PREFIX':'amlas-test','AMLAS_SSM_PREFIX':'/amlas'})
        self.assertNotEqual(x.returncode,0)
        self.assertIn('ssm',x.stderr.lower())
    def test_entrypoint_accepts_namespace_selector_and_documents_it(self):
        x=subprocess.run(['bash','scripts/setup.sh','--deployment-prefix','amlas-fresh-1013','--help'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(x.returncode,0,x.stdout+x.stderr)
        self.assertIn('--deployment-prefix',x.stdout)

if __name__=='__main__':
    unittest.main()
