"""Tests for the audio summarization feature."""

from unittest.mock import MagicMock, patch, mock_open
import pytest
from click.testing import CliRunner

from summarize import main, transcribe_audio, summarize_text, SUPPORTED_FORMATS


class TestSupportedFormats:
    def test_supported_formats_includes_common_audio(self):
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".wav" in SUPPORTED_FORMATS
        assert ".m4a" in SUPPORTED_FORMATS
        assert ".webm" in SUPPORTED_FORMATS
        assert ".ogg" in SUPPORTED_FORMATS
        assert ".flac" in SUPPORTED_FORMATS


class TestTranscribeAudio:
    def test_calls_whisper_api_with_correct_params(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = MagicMock(text="Hello world")

        result = transcribe_audio(mock_client, str(audio_file))

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-1"

    def test_returns_transcript_text(self, tmp_path):
        audio_file = tmp_path / "speech.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = MagicMock(
            text="Bu bir test metnidir."
        )

        result = transcribe_audio(mock_client, str(audio_file))
        assert result == "Bu bir test metnidir."


class TestSummarizeText:
    def test_turkish_summary_uses_turkish_prompts(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Özet: Test"))]
        )

        result = summarize_text(mock_client, "Uzun bir metin", language="tr")

        assert result == "Özet: Test"
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert any("özetle" in m["content"].lower() for m in messages)

    def test_english_summary_uses_english_prompts(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Summary: Test"))]
        )

        result = summarize_text(mock_client, "A long text", language="en")

        assert result == "Summary: Test"
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert any("summarize" in m["content"].lower() for m in messages)

    def test_returns_summary_content(self):
        mock_client = MagicMock()
        expected = "This is the summary."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=expected))]
        )

        result = summarize_text(mock_client, "Some text", language="en")
        assert result == expected


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_missing_api_key_shows_error(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake")

        result = self.runner.invoke(main, [str(audio_file)], env={"OPENAI_API_KEY": ""})
        assert result.exit_code != 0
        combined = result.output + (str(result.exception) if result.exception else "")
        assert "api key" in combined.lower()

    def test_unsupported_format_shows_error(self, tmp_path):
        bad_file = tmp_path / "audio.xyz"
        bad_file.write_bytes(b"fake")

        result = self.runner.invoke(main, [str(bad_file), "--api-key", "test-key"])
        assert result.exit_code != 0
        assert "unsupported" in result.output.lower()

    def test_successful_run_prints_summary(self, tmp_path):
        audio_file = tmp_path / "speech.mp3"
        audio_file.write_bytes(b"fake audio data")

        with patch("summarize.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = MagicMock(
                text="Test transcript text."
            )
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test summary."))]
            )

            result = self.runner.invoke(
                main, [str(audio_file), "--api-key", "fake-key"]
            )

        assert result.exit_code == 0
        assert "Test summary." in result.output

    def test_transcript_flag_prints_transcript(self, tmp_path):
        audio_file = tmp_path / "speech.wav"
        audio_file.write_bytes(b"fake audio data")

        with patch("summarize.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = MagicMock(
                text="Full transcript text."
            )
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="Short summary."))]
            )

            result = self.runner.invoke(
                main, [str(audio_file), "--api-key", "fake-key", "--transcript"]
            )

        assert result.exit_code == 0
        assert "Full transcript text." in result.output
        assert "Short summary." in result.output

    def test_english_language_option(self, tmp_path):
        audio_file = tmp_path / "speech.mp3"
        audio_file.write_bytes(b"fake audio data")

        with patch("summarize.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.audio.transcriptions.create.return_value = MagicMock(
                text="Some spoken words."
            )
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="English summary."))]
            )

            result = self.runner.invoke(
                main, [str(audio_file), "--api-key", "fake-key", "--language", "en"]
            )

        assert result.exit_code == 0
        assert "English summary." in result.output
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert any("summarize" in m["content"].lower() for m in messages)
