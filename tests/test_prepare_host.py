"""Closed-boundary host preparation tests; stdlib runner works on legacy builders."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/lib/prepare_host.sh"

TOOL = r"""#!/usr/bin/python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
state = pathlib.Path(os.environ['TEST_STATE'])
with (state/'calls.jsonl').open('a') as f:
    f.write(json.dumps({'tool':name,'args':args,'docker_config':os.environ.get('DOCKER_CONFIG')})+'\n')
ready = (state/'ready').exists()
if name == 'sudo':
    if os.environ.get('TEST_SUDO_DENY') == '1': sys.exit(1)
    assert args and args.pop(0) == '-n', 'sudo must be noninteractive'
    if args and args[0] == '--': args.pop(0)
    assert args[0] in ('true','dnf','systemctl','env','/usr/bin/env') or args[0].endswith('/docker'), args
    env = dict(os.environ, TEST_ROOT='1')
    os.execvpe(args[0], args, env)
if name == 'dnf':
    allowed = {'docker','nodejs22','nodejs22-npm','cmake','git','python3.11','python3.11-devel','gcc','gcc-c++','make','zip','unzip','tar','gzip','curl','gnupg2'}
    assert args[:2] == ['install','-y'], args
    assert args[2:] and set(args[2:]) <= allowed, args
    (state/'ready').touch()
    sys.exit(0)
if name == 'systemctl':
    assert args == ['start','docker'], args
    (state/'daemon').touch()
    sys.exit(0)
if name == 'aws':
    assert args == ['--version'], 'host preparation must never call AWS services'
    print('aws-cli/2.36.47 Python/3.13 Linux/aarch64')
    sys.exit(0)
if name == 'docker':
    if args in (['--version'], ['version','--format','{{.Client.Version}}']):
        print('Docker version 25.0.16'); sys.exit(0)
    if args == ['info']:
        sys.exit(0 if ready and (state/'daemon').exists() and os.environ.get('TEST_ROOT') == '1' else 1)
    if args == ['buildx','version']:
        if ready and os.environ.get('TEST_BUILDX_ABSENT') != '1': print('github.com/docker/buildx v0.38.0'); sys.exit(0)
        sys.exit(1)
    sys.exit(91)
if name == 'curl':
    assert '--proto' in args and '=https' in args and '--tlsv1.2' in args, args
    target = args[args.index('-o')+1]
    pathlib.Path(target).write_text('unsigned, untrusted fixture download\n')
    sys.exit(0)
if name == 'uname':
    assert args in (['-m'],['-s'])
    print('aarch64' if args == ['-m'] else 'Linux'); sys.exit(0)
if name in ('node','npm','cmake','gcc','g++','make','git','zip','python3.11','unzip','gpg'):
    if not ready: sys.exit(127)
    print('v22.23.2' if name == 'node' else '3.11.16' if name == 'python3.11' else 'tool-version')
    sys.exit(0)
raise AssertionError((name,args))
"""


class HostPreparationTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(os.environ.get("TMPDIR", str(ROOT.parent / "tmp")))
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=str(temp_root))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.state = self.root / "state"
        self.state.mkdir()
        self.tmp = self.root / "tmp"
        self.tmp.mkdir()
        self.env = {
            "PATH": str(self.bin),
            "TMPDIR": str(self.tmp),
            "TEST_STATE": str(self.state),
            "LANG": "C",
            "HOME": str(self.root / "home"),
        }
        tool = self.bin / "fixture-tool"
        tool.write_text(TOOL.replace("#!/usr/bin/python3", "#!" + sys.executable, 1))
        tool.chmod(0o755)
        for name in (
            "sudo",
            "dnf",
            "systemctl",
            "aws",
            "docker",
            "curl",
            "uname",
            "node",
            "npm",
            "cmake",
            "gcc",
            "g++",
            "make",
            "git",
            "zip",
            "python3.11",
            "unzip",
            "gpg",
        ):
            (self.bin / name).symlink_to(tool)
        for name in (
            "bash",
            "mkdir",
            "cat",
            "awk",
            "chmod",
            "dirname",
            "sha256sum",
            "mv",
            "cp",
            "id",
            "env",
            "true",
            "sed",
            "head",
            "tail",
            "readlink",
            "ln",
            "touch",
        ):
            if not (self.bin / name).exists():
                (self.bin / name).symlink_to(shutil.which(name))

    def run_shell(self, code, **extra):
        env = dict(self.env, **extra)
        return subprocess.run(
            ["/bin/bash", "-uc", 'source "$1"; ' + code, "test-host", str(HELPER)],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def calls(self):
        path = self.state / "calls.jsonl"
        return [] if not path.exists() else [json.loads(s) for s in path.read_text().splitlines()]

    def test_source_is_side_effect_free(self):
        result = self.run_shell(":")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.calls(), [])
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_check_reports_missing_without_installing(self):
        result = subprocess.run(
            ["/bin/bash", str(HELPER), "--check"],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("node", (result.stdout + result.stderr).lower())
        self.assertFalse(any(c["tool"] in ("curl", "dnf", "sudo", "systemctl") for c in self.calls()))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_al2023_installs_prerequisites_and_uses_local_npm_docker_config(self):
        result = self.run_shell(
            '_amlas_host_platform(){ printf "amzn:2023\\n"; }; prepare_host_prerequisites; printf "NPM=%s\\nDOCKER_CONFIG=%s\\n" "$npm_config_prefix" "$DOCKER_CONFIG"; docker info'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installs = [c for c in self.calls() if c["tool"] == "dnf"]
        self.assertEqual(len(installs), 1, self.calls())
        self.assertTrue(
            {"docker", "nodejs22", "nodejs22-npm", "cmake", "git", "python3.11", "python3.11-devel"}
            <= set(installs[0]["args"])
        )
        self.assertIn("NPM=" + str(self.tmp), result.stdout)
        self.assertIn("DOCKER_CONFIG=" + str(self.tmp), result.stdout)
        infos = [c for c in self.calls() if c["tool"] == "docker" and c["args"] == ["info"]]
        self.assertTrue(infos)
        self.assertTrue(infos[-1]["docker_config"].startswith(str(self.tmp)))
        self.assertFalse(any("usermod" in c["args"] or "chmod" in c["args"] for c in self.calls()))

    def test_sudo_denial_fails_without_install_or_cache_writes(self):
        result = self.run_shell(
            '_amlas_host_platform(){ printf "amzn:2023\\n"; }; prepare_host_prerequisites', TEST_SUDO_DENY="1"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sudo", (result.stdout + result.stderr).lower())
        self.assertFalse(any(c["tool"] in ("dnf", "curl", "systemctl") for c in self.calls()))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_legacy_al2_gives_container_remediation_without_mutation(self):
        result = self.run_shell('_amlas_host_platform(){ printf "amzn:2\\n"; }; prepare_host_prerequisites')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("container", (result.stdout + result.stderr).lower())
        self.assertFalse(any(c["tool"] in ("sudo", "dnf", "curl", "systemctl") for c in self.calls()))

    def test_existing_aws_is_preserved_before_identity(self):
        result = self.run_shell("ensure_aws_cli_prerequisite")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(c["tool"] in ("curl", "sudo", "dnf") for c in self.calls()))

    def test_tampered_buildx_download_is_rejected(self):
        result = self.run_shell(
            '_amlas_host_platform(){ printf "amzn:2023\\n"; }; prepare_host_prerequisites',
            TEST_BUILDX_ABSENT="1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any(c["tool"] == "curl" for c in self.calls()), result.stdout + result.stderr)
        self.assertFalse(list(self.tmp.glob("**/docker-buildx")))

    def test_tampered_aws_installer_never_executes(self):
        (self.bin / "aws").unlink()
        result = self.run_shell("ensure_aws_cli_prerequisite")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any(c["tool"] == "curl" for c in self.calls()), result.stdout + result.stderr)
        self.assertIn("checksum", (result.stdout + result.stderr).lower())


if __name__ == "__main__":
    unittest.main()
