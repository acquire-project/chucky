import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from manifest import verify_corpus
from test_manifest import fixture, sha


def git(root, *args):
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {args}: {result.stderr}")
    return result.stdout.strip()


@unittest.skipUnless(shutil.which("git-annex"), "git-annex is not installed")
class StorageTests(unittest.TestCase):
    def test_annex_clone_and_plain_export_have_identical_pixels(self):
        with tempfile.TemporaryDirectory(prefix="chucky-storage-") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            document = fixture(source)
            notice = b"Fixture creator. Share under the fixture license.\n"
            (source / "NOTICE.md").write_bytes(notice)
            document["notices"] = [{"path": "NOTICE.md", "sha256": sha(notice)}]
            survey_data = b'{"kind": "synthetic-test", "status": "complete"}\n'
            (source / "survey.json").write_bytes(survey_data)
            document["selection"] = {
                "survey_path": "survey.json",
                "survey_sha256": sha(survey_data),
            }
            (source / "manifest.json").write_text(json.dumps(document))
            (source / ".gitattributes").write_text(
                "* text=auto eol=lf annex.largefiles=nothing\n"
                "*.raw -text annex.largefiles=anything annex.backend=SHA256\n"
            )
            git(source, "init", "--initial-branch=main")
            git(source, "config", "user.name", "Storage test")
            git(source, "config", "user.email", "storage@example.invalid")
            git(source, "annex", "init", "source-test")
            packs = ["pack-0.raw", "pack-1.raw"]
            git(source, "annex", "add", "--backend=SHA256", "--", *packs)
            git(
                source,
                "add",
                ".gitattributes",
                "manifest.json",
                "evidence.txt",
                "survey.json",
                "NOTICE.md",
            )
            git(source, "commit", "-m", "bench: storage test")
            original = verify_corpus(source, allow_test_data=True)
            self.assertTrue(
                all(
                    "/annex/objects/" in path.as_posix()
                    for path in original.pack_files.values()
                )
            )
            for pack in packs:
                self.assertTrue(
                    git(source, "annex", "lookupkey", pack).startswith("SHA256-")
                )
            clone = root / "clone"
            git(root, "clone", "--no-hardlinks", str(source), str(clone))
            git(clone, "annex", "init", "clone-test")
            with self.assertRaisesRegex(ValueError, "git annex get"):
                verify_corpus(clone, allow_test_data=True)
            git(clone, "annex", "get", "--", *packs)
            git(clone, "annex", "fsck", "--", *packs)
            copied = verify_corpus(clone, allow_test_data=True)
            self.assertTrue(
                all(
                    "/annex/objects/" in path.as_posix()
                    for path in copied.pack_files.values()
                )
            )
            superproject = root / "superproject"
            superproject.mkdir()
            git(superproject, "init", "--initial-branch=main")
            git(superproject, "config", "user.name", "Storage test")
            git(superproject, "config", "user.email", "storage@example.invalid")
            git(
                superproject,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                str(source),
                "data",
            )
            submodule = superproject / "data"
            git(submodule, "annex", "init", "submodule-test")
            with self.assertRaisesRegex(ValueError, "git annex get"):
                verify_corpus(submodule, allow_test_data=True)
            git(submodule, "annex", "get", "--", *packs)
            submodule_corpus = verify_corpus(submodule, allow_test_data=True)
            annex_objects = (
                superproject / ".git" / "modules" / "data" / "annex" / "objects"
            ).resolve()
            self.assertTrue(
                all(
                    path.is_relative_to(annex_objects)
                    for path in submodule_corpus.pack_files.values()
                )
            )
            archive = root / "corpus.zip"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("export.py")),
                    "--direct-corpus",
                    str(source),
                    "--output",
                    str(archive),
                    "--allow-test-data",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plain = root / "plain"
            with zipfile.ZipFile(archive) as exported:
                exported.extractall(plain)
            materialized = verify_corpus(plain, allow_test_data=True)
            self.assertEqual(original.sha256, copied.sha256)
            self.assertEqual(original.sha256, materialized.sha256)
            self.assertEqual((clone / "NOTICE.md").read_bytes(), notice)
            self.assertEqual((plain / "NOTICE.md").read_bytes(), notice)
            for pack in packs:
                self.assertFalse((plain / pack).is_symlink())
                self.assertEqual(
                    (source / pack).read_bytes(), (clone / pack).read_bytes()
                )
                self.assertEqual(
                    (source / pack).read_bytes(), (plain / pack).read_bytes()
                )


if __name__ == "__main__":
    unittest.main()
