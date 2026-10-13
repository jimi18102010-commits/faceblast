import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from faceblast import cli, models
from faceblast.recognizer import MatchResult
from faceblast.ui import print_match_result

@pytest.mark.parametrize('draw,expected', [('y', True), ('n', False)])
def test_interactive_detect_args(draw, expected):
    with patch.object(cli.Prompt, 'ask', side_effect=['2', 'face.jpg', draw]), patch.object(cli, 'handle_detect') as detect, patch.object(cli, 'print_banner'):
        cli.handle_interactive()
    args = detect.call_args.args[0]
    assert args.image == 'face.jpg'
    assert args.draw is expected
    assert args.threshold == 0.6

@pytest.mark.parametrize('is_match', [True, False])
def test_custom_threshold_display(is_match):
    result = MatchResult(is_match, 0.7, 0.5, True, True)
    with patch('faceblast.ui.console.print') as output:
        print_match_result(result, 'a', 'b', threshold=0.8)
    assert '0.8000' in output.call_args.args[0].renderable


def mock_urlopen(payload):
    response = io.BytesIO(payload)
    response.headers = {'content-length': str(len(payload))}
    return response


def test_model_download_verified(tmp_path):
    payload = b'trusted-model'; dest = tmp_path / 'model.onnx'
    with patch.object(models.urllib.request, 'urlopen', return_value=mock_urlopen(payload)) as open_url:
        models.download_file_with_progress('https://example.test/model', dest, hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload
    assert open_url.call_args.kwargs['timeout'] == 30
    assert list(tmp_path.glob('*.tmp')) == []


def test_bad_model_cannot_replace_cached_copy(tmp_path):
    dest = tmp_path / 'model.onnx'; dest.write_bytes(b'old-good-model')
    with patch.object(models.urllib.request, 'urlopen', return_value=mock_urlopen(b'corrupt')):
        with pytest.raises(ValueError, match='SHA-256'):
            models.download_file_with_progress('https://example.test/model', dest, '0' * 64)
    assert dest.read_bytes() == b'old-good-model'
    assert list(tmp_path.glob('*.tmp')) == []


def test_corrupt_cache_redownloaded(tmp_path):
    (tmp_path / models.YUNET_FILENAME).write_bytes(b'corrupt')
    with patch.object(models, 'CACHE_DIR', tmp_path), patch.object(models, 'download_file_with_progress') as download:
        models.get_detector_model_path()
    download.assert_called_once()
    assert download.call_args.args[2] == models.YUNET_SHA256


def test_urls_are_pinned():
    assert '/main/' not in models.YUNET_URL
    assert models.MODEL_COMMIT in models.YUNET_URL
    assert models.MODEL_COMMIT in models.SFACE_URL
