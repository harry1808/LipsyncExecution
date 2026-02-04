from webapp import create_app

app = create_app()

if __name__ == "__main__":
    # use_reloader=False prevents Flask from restarting during long-running tasks
    # like Wav2Lip lip sync processing
    app.run(debug=True, use_reloader=False)

