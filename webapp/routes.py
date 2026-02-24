import logging
import shutil
from pathlib import Path
from uuid import uuid4

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from . import db
from .dubbing import process_video
from .language_support import SUPPORTED_LANGS
from .models import Activity
from .evaluation_metrics import calculate_bleu, calculate_wer, calculate_cer, calculate_composite_score
from .evaluate_dubbing import DubbingEvaluator
from . import lipsync_metrics

main_bp = Blueprint("main", __name__)
VOICE_CHOICES = ("female", "male")


@main_bp.route("/")
def landing():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return render_template(
        "landing.html",
        languages=SUPPORTED_LANGS,
        hero_bg=current_app.config.get("HERO_BACKGROUND"),
    )


@main_bp.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        return _handle_video_submission()

    activities = (
        Activity.query.filter_by(user_id=current_user.id)
        .order_by(Activity.created_at.desc())
        .limit(10)
        .all()
    )
    return render_template(
        "dashboard.html",
        activities=activities,
        languages=SUPPORTED_LANGS,
        voice_choices=VOICE_CHOICES,
        lipsync_default=current_app.config.get("LIPSYNC_DEFAULT", False),
        hero_bg=current_app.config.get("HERO_BACKGROUND"),
    )


def _handle_video_submission():
    file = request.files.get("video_file")
    source_lang = request.form.get("source_lang", "en").lower()
    dest_lang = request.form.get("dest_lang", "fr").lower()
    voice = request.form.get("voice", VOICE_CHOICES[0]).lower()
    enable_lipsync = request.form.get("enable_lipsync") == "on"

    if not file or file.filename == "":
        flash("Please choose a video file before submitting.", "warning")
        return redirect(url_for("main.dashboard"))

    if source_lang == dest_lang:
        flash("Source and target languages must be different.", "warning")
        return redirect(url_for("main.dashboard"))

    if voice not in VOICE_CHOICES:
        flash("Please select a valid voice option.", "warning")
        return redirect(url_for("main.dashboard"))

    filename = secure_filename(file.filename) or "upload.mp4"
    unique_name = f"{uuid4().hex}_{filename}"
    upload_path = Path(current_app.config["UPLOAD_FOLDER"]) / unique_name
    file.save(upload_path)

    activity = Activity(
        user_id=current_user.id,
        filename=filename,
        source_lang=source_lang,
        dest_lang=dest_lang,
        status="processing",
    )
    db.session.add(activity)
    db.session.commit()

    original_storage_name = None

    try:
        # Process video with Wav2Lip lip-sync if enabled
        final_path, transcript, translation = process_video(
            upload_path,
            source_lang,
            dest_lang,
            current_app.config["OUTPUT_FOLDER"],
            current_app.logger,
            voice=voice,
            enable_lipsync=enable_lipsync or current_app.config.get("LIPSYNC_DEFAULT", False),
            lipsync_assets_dir=current_app.config.get("WAV2LIP_ASSETS_DIR"),
        )
        suffix = upload_path.suffix or ".mp4"
        original_storage_name = f"original_{uuid4().hex}{suffix}"
        original_storage_path = Path(current_app.config["OUTPUT_FOLDER"]) / original_storage_name
        shutil.copy(upload_path, original_storage_path)

        activity.mark_finished(
            final_path.name,
            transcript,
            translation,
            original_filename=original_storage_name,
        )
        db.session.commit()
        flash("Your video has been dubbed successfully!", "success")
    except Exception as exc:  # pragma: no cover - best-effort logging
        current_app.logger.exception("Video processing failed.")
        db.session.rollback()
        persisted = Activity.query.get(activity.id)
        if persisted:
            persisted.mark_failed(str(exc))
            db.session.commit()
        flash("We could not process that video. Check the activity feed for details.", "danger")
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            current_app.logger.log(logging.WARNING, "Could not delete temporary upload %s", upload_path)

    return redirect(url_for("main.dashboard"))


def _remove_activity_files(activity):
    output_folder = Path(current_app.config["OUTPUT_FOLDER"])
    for filename in (activity.output_filename, activity.stored_original_filename):
        if not filename:
            continue
        file_path = output_folder / filename
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            current_app.logger.log(
                logging.WARNING, "Could not delete file %s", file_path
            )


@main_bp.route("/activity/<int:activity_id>")
@login_required
def activity_detail(activity_id):
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id:
        abort(403)
    return render_template("activity_detail.html", activity=activity)


@main_bp.route("/download/<int:activity_id>")
@login_required
def download(activity_id):
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id or activity.status != "completed":
        abort(403)

    output_folder = Path(current_app.config["OUTPUT_FOLDER"])
    if not activity.output_filename:
        abort(404)
    file_path = output_folder / activity.output_filename
    if not file_path.exists():
        abort(404)

    return send_from_directory(
        output_folder,
        activity.output_filename,
        as_attachment=True,
        download_name=f"dubbed_{activity.filename}",
    )


@main_bp.route("/media/<int:activity_id>")
@login_required
def stream_video(activity_id):
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id or activity.status != "completed":
        abort(403)

    if not activity.output_filename:
        abort(404)

    output_folder = Path(current_app.config["OUTPUT_FOLDER"])
    file_path = output_folder / activity.output_filename
    if not file_path.exists():
        abort(404)

    return send_from_directory(
        output_folder,
        activity.output_filename,
        as_attachment=False,
        download_name=f"dubbed_{activity.filename}",
    )


@main_bp.route("/media/original/<int:activity_id>")
@login_required
def stream_original_video(activity_id):
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id or activity.status != "completed":
        abort(403)

    if not activity.stored_original_filename:
        abort(404)

    output_folder = Path(current_app.config["OUTPUT_FOLDER"])
    file_path = output_folder / activity.stored_original_filename
    if not file_path.exists():
        abort(404)

    return send_from_directory(
        output_folder,
        activity.stored_original_filename,
        as_attachment=False,
        download_name=activity.filename,
    )


@main_bp.route("/activity/<int:activity_id>/delete", methods=["POST"])
@login_required
def delete_activity(activity_id):
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id:
        abort(403)

    _remove_activity_files(activity)
    db.session.delete(activity)
    db.session.commit()
    flash("Activity deleted successfully.", "info")
    return redirect(url_for("main.dashboard"))


@main_bp.route("/evaluate")
@login_required
def evaluation_page():
    """Evaluation page - list all activities that can be evaluated."""
    activities = (
        Activity.query.filter_by(user_id=current_user.id, status="completed")
        .order_by(Activity.created_at.desc())
        .all()
    )
    return render_template(
        "evaluation.html",
        activities=activities,
        languages=SUPPORTED_LANGS,
    )


@main_bp.route("/evaluate/<int:activity_id>/helper", methods=["GET"])
@login_required
def evaluation_helper(activity_id):
    """Helper page to create better ground truth by comparing translations."""
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id:
        abort(403)
    
    if activity.status != "completed":
        flash("Can only evaluate completed activities.", "warning")
        return redirect(url_for("main.evaluation_page"))
    
    return render_template(
        "evaluation_helper.html",
        activity=activity,
        languages=SUPPORTED_LANGS,
    )


@main_bp.route("/evaluate/<int:activity_id>", methods=["GET", "POST"])
@login_required
def evaluate_activity(activity_id):
    """Evaluate a specific activity with ground truth data."""
    activity = Activity.query.get_or_404(activity_id)
    if activity.user_id != current_user.id:
        abort(403)
    
    if activity.status != "completed":
        flash("Can only evaluate completed activities.", "warning")
        return redirect(url_for("main.evaluation_page"))
    
    if request.method == "POST":
        # Get ground truth from form
        ground_truth_transcript = request.form.get("ground_truth_transcript", "").strip()
        ground_truth_translation = request.form.get("ground_truth_translation", "").strip()
        
        if not ground_truth_transcript or not ground_truth_translation:
            flash("Please provide both ground truth transcript and translation.", "warning")
            return render_template(
                "evaluate_form.html",
                activity=activity,
                languages=SUPPORTED_LANGS,
            )
        
        # Calculate evaluation metrics
        try:
            # ASR Evaluation (if we have transcript)
            asr_metrics = None
            if activity.transcript:
                wer_result = calculate_wer(ground_truth_transcript, activity.transcript)
                cer_result = calculate_cer(ground_truth_transcript, activity.transcript)
                asr_metrics = {
                    'wer': wer_result['wer'],
                    'cer': cer_result['cer'],
                    'accuracy': wer_result['accuracy'],
                    'substitutions': wer_result['substitutions'],
                    'deletions': wer_result['deletions'],
                    'insertions': wer_result['insertions'],
                    'ground_truth': ground_truth_transcript,
                    'hypothesis': activity.transcript
                }
            
            # Translation Evaluation (if we have translation)
            translation_metrics = None
            if activity.translated_text:
                bleu_result = calculate_bleu(activity.translated_text, ground_truth_translation)
                translation_metrics = {
                    'bleu_score': bleu_result['bleu_score'],
                    'bleu_1': bleu_result['bleu_1'],
                    'bleu_2': bleu_result['bleu_2'],
                    'bleu_3': bleu_result['bleu_3'],
                    'bleu_4': bleu_result['bleu_4'],
                    'brevity_penalty': bleu_result['brevity_penalty'],
                    'ground_truth': ground_truth_translation,
                    'hypothesis': activity.translated_text
                }
            
            # Lip-sync evaluation (when output video exists; uses Wav2Lip SyncNet when assets available)
            lipsync_result = None
            output_folder = Path(current_app.config["OUTPUT_FOLDER"])
            if activity.output_filename:
                output_video_path = output_folder / activity.output_filename
                if output_video_path.exists():
                    try:
                        lipsync_result = lipsync_metrics.evaluate_lipsync(
                            video_path=str(output_video_path),
                            run_syncnet=True,
                            wav2lip_assets_dir=current_app.config.get("WAV2LIP_ASSETS_DIR"),
                            logger=current_app.logger,
                        )
                    except Exception as e:
                        current_app.logger.warning("Lip-sync evaluation failed (non-fatal): %s", e)
            
            # Build composite metrics and calculate overall score (same formula as pipeline evaluator)
            composite_metrics = {}
            if asr_metrics:
                composite_metrics["wer"] = asr_metrics["wer"]
            if translation_metrics:
                composite_metrics["bleu_score"] = translation_metrics["bleu_score"]
            if lipsync_result and lipsync_result.get("lipsync_score") is not None:
                composite_metrics["lipsync_score"] = lipsync_result["lipsync_score"]
            composite_score = calculate_composite_score(composite_metrics)
            
            # Determine rating
            if composite_score >= 90:
                rating = "Excellent"
                stars = 5
            elif composite_score >= 75:
                rating = "Good"
                stars = 4
            elif composite_score >= 60:
                rating = "Fair"
                stars = 3
            elif composite_score >= 40:
                rating = "Poor"
                stars = 2
            else:
                rating = "Needs Improvement"
                stars = 1
            
            results = {
                'asr': asr_metrics,
                'translation': translation_metrics,
                'lipsync': lipsync_result,
                'composite_score': composite_score,
                'rating': rating,
                'stars': stars
            }
            
            return render_template(
                "evaluation_results.html",
                activity=activity,
                results=results,
                languages=SUPPORTED_LANGS,
            )
            
        except Exception as e:
            current_app.logger.exception("Evaluation failed")
            flash(f"Evaluation failed: {str(e)}", "danger")
            return redirect(url_for("main.evaluation_page"))
    
    # GET request - show form
    return render_template(
        "evaluate_form.html",
        activity=activity,
        languages=SUPPORTED_LANGS,
    )

