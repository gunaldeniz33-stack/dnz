"""Sesten özet çıkar - Extract summary from audio files."""

import os
import sys
import click
from openai import OpenAI

SUPPORTED_FORMATS = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac")


def transcribe_audio(client: OpenAI, audio_path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper."""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcript.text


def summarize_text(client: OpenAI, text: str, language: str = "tr") -> str:
    """Summarize a text using OpenAI GPT."""
    if language == "tr":
        system_prompt = "Sen bir metin özetleme asistanısın. Verilen metni kısa ve öz bir şekilde özetle."
        user_prompt = f"Aşağıdaki metni özetle:\n\n{text}"
    else:
        system_prompt = "You are a text summarization assistant. Summarize the given text concisely."
        user_prompt = f"Summarize the following text:\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return content if content is not None else ""


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, readable=True))
@click.option(
    "--language",
    "-l",
    default="tr",
    show_default=True,
    help="Summary output language: 'tr' for Turkish, 'en' for English.",
)
@click.option(
    "--transcript",
    "-t",
    is_flag=True,
    default=False,
    help="Also print the full transcript.",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (defaults to OPENAI_API_KEY environment variable).",
)
def main(audio_file: str, language: str, transcript: bool, api_key: str) -> None:
    """Extract a summary from an audio file.

    AUDIO_FILE is the path to an audio file to transcribe and summarize.
    Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg, flac.
    """
    ext = os.path.splitext(audio_file)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise click.BadParameter(
            f"Unsupported file format '{ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            param_hint="AUDIO_FILE",
        )

    if not api_key:
        raise click.UsageError(
            "OpenAI API key is required. Set the OPENAI_API_KEY environment variable "
            "or pass it with --api-key."
        )

    client = OpenAI(api_key=api_key)

    click.echo("🎙️  Transcribing audio...", err=True)
    text = transcribe_audio(client, audio_file)

    if transcript:
        click.echo("\n--- Transcript ---")
        click.echo(text)
        click.echo("--- End of Transcript ---\n")

    click.echo("📝  Summarizing...", err=True)
    summary = summarize_text(client, text, language=language)

    click.echo(summary)


if __name__ == "__main__":
    main()
