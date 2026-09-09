from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import utils  # Your custom module
from utils import preprocess_and_save

import json
import re
import secrets
import time
import hashlib
import smtplib
from email.message import EmailMessage


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ----------------------
import seaborn as sns
import numpy as np
import io
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import analysis_tools

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")  # Change in production!

os.makedirs(os.path.join(app.root_path, "static"), exist_ok=True)
UPLOADS_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(app.root_path, "users.db")


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Persist history only in users.chat_history (single "full chat history" field).
    # Remove table from earlier implementation if present.
    conn.execute("DROP TABLE IF EXISTS search_history")

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "email" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    if "chat_history" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN chat_history TEXT NOT NULL DEFAULT '[]'")
    conn.commit()
    conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


init_db()

def _save_uploaded_file(file_storage):
    if not file_storage or not getattr(file_storage, "filename", ""):
        return None, "No file provided."
    filename = secure_filename(file_storage.filename)
    if not filename:
        return None, "Invalid filename."
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in {".csv", ".json", ".xlsx", ".xls"}:
        return None, "Unsupported file type."

    token = secrets.token_hex(8)
    saved_name = f"{int(time.time())}_{token}_{filename}"
    saved_path = os.path.join(UPLOADS_DIR, saved_name)
    file_storage.save(saved_path)
    return saved_path, ""


def _load_df_from_path(path: str):
    if not path or not os.path.exists(path):
        return None, [], "", "Saved file not found. Please upload again."
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            return None, [], "", "Unsupported saved file type."
        df.dropna(how="all", inplace=True)
        cols = df.columns.tolist()
        return df, cols, "", ""
    except Exception as e:
        return None, [], "", str(e)

def load_user_chat_history(user_id: int):
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT chat_history FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        raw = (row["chat_history"] if row else None) or "[]"
        try:
            data = json.loads(raw)
        except Exception:
            data = []
        return data if isinstance(data, list) else []
    finally:
        conn.close()


def save_user_chat_history(user_id: int, chat_history) -> None:
    # last 5 searches => last 10 messages (user + assistant)
    if not isinstance(chat_history, list):
        chat_history = []
    trimmed = chat_history[-10:]
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE users SET chat_history = ? WHERE id = ?",
            (json.dumps(trimmed, ensure_ascii=False), user_id),
        )
        conn.commit()
    finally:
        conn.close()

def _is_valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (email or "").strip().lower()))


def _hash_otp(email: str, otp: str, salt: str) -> str:
    payload = f"{email.strip().lower()}|{otp.strip()}|{salt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _send_otp_email(to_email: str, otp: str) -> None:
    """
    Sends OTP via SMTP. Configure these env vars:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM(optional)
    If SMTP_HOST is missing, OTP is printed to server console (dev fallback).
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_from = os.getenv("SMTP_FROM") or smtp_user

    if not smtp_host or not smtp_user or not smtp_pass or not smtp_from:
        # Dev fallback: no email configuration
        print(f"[DEV OTP] Email={to_email} OTP={otp}")
        return

    msg = EmailMessage()
    msg["Subject"] = "Your verification OTP"
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(f"Your OTP for verification is: {otp}\n\nIt expires in 10 minutes.")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    except Exception as e:
        # Make the root cause visible during development
        print(f"[SMTP ERROR] {type(e).__name__}: {e}")
        raise


@app.route("/clear", methods=["GET"])
@login_required
def clear():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

@app.route("/clear-file", methods=["GET"])
@login_required
def clear_file():
    path = session.pop("uploaded_file_path", None)
    session.pop("uploaded_file_name", None)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""
    if request.method == "POST":
        identifier = (request.form.get("identifier") or "").strip()
        password = request.form.get("password") or ""

        if not identifier or not password:
            message = "Please enter username/email and password."
        else:
            conn = get_db_connection()
            if "@" in identifier:
                email = identifier.strip().lower()
                user = conn.execute(
                    "SELECT id, username, password_hash, email FROM users WHERE email = ?",
                    (email,),
                ).fetchone()
            else:
                username = identifier
                user = conn.execute(
                    "SELECT id, username, password_hash, email FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
            conn.close()

            if not user or not check_password_hash(user["password_hash"], password):
                message = "Invalid username/email or password."
            else:
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["chat_history"] = load_user_chat_history(user["id"])
                return redirect(url_for("index"))

    return render_template("login.html", message=message)


@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not email or not password:
            message = "Username, email, and password are required."
        elif not _is_valid_email(email):
            message = "Please enter a valid email address."
        elif len(password) < 6:
            message = "Password must be at least 6 characters long."
        elif password != confirm_password:
            message = "Passwords do not match."
        else:
            # Ensure username/email aren't already taken before sending OTP
            conn = get_db_connection()
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username = ? OR email = ? LIMIT 1",
                (username, email),
            ).fetchone()
            conn.close()
            if exists:
                message = "Username or email already exists. Please choose another."
            else:
                otp = f"{secrets.randbelow(1000000):06d}"
                salt = secrets.token_hex(16)
                session["pending_registration"] = {
                    "username": username,
                    "email": email,
                    "password_hash": generate_password_hash(password),
                    "otp_hash": _hash_otp(email, otp, salt),
                    "otp_salt": salt,
                    "otp_expires_at": int(time.time()) + 600,
                    "otp_attempts": 0,
                }
                session.modified = True
                try:
                    _send_otp_email(email, otp)
                except Exception as e:
                    message = f"Failed to send OTP email: {type(e).__name__}: {str(e)}"
                else:
                    return redirect(url_for("verify_email"))

    return render_template("register.html", message=message)


@app.route("/verify-email", methods=["GET", "POST"])
def verify_email():
    message = ""
    pending = session.get("pending_registration")
    if not pending:
        return redirect(url_for("register"))

    if request.method == "POST":
        otp = (request.form.get("otp") or "").strip()
        now = int(time.time())
        if now > int(pending.get("otp_expires_at") or 0):
            session.pop("pending_registration", None)
            return render_template("verify_email.html", message="OTP expired. Please register again.")

        pending["otp_attempts"] = int(pending.get("otp_attempts") or 0) + 1
        session["pending_registration"] = pending
        session.modified = True

        if pending["otp_attempts"] > 5:
            session.pop("pending_registration", None)
            return render_template("verify_email.html", message="Too many attempts. Please register again.")

        expected = pending.get("otp_hash") or ""
        salt = pending.get("otp_salt") or ""
        email = pending.get("email") or ""
        if not otp or _hash_otp(email, otp, salt) != expected:
            message = "Invalid OTP. Please try again."
        else:
            conn = get_db_connection()
            try:
                conn.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    (pending["username"], pending["email"], pending["password_hash"]),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                message = "Username or email already exists. Please register again."
            finally:
                conn.close()

            if not message:
                session.pop("pending_registration", None)
                return redirect(url_for("login"))

    return render_template("verify_email.html", message=message, email=pending.get("email"))


@app.route("/logout", methods=["GET"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/search", methods=["GET", "POST"])
@login_required
def web_search():
    message = ""
    response = ""
    if "chat_history" not in session:
        session["chat_history"] = load_user_chat_history(session["user_id"])

    if request.method == "POST":
        question = (request.form.get("question", "") or "").strip()
        groq_key = request.form.get("api_key") or os.getenv("GROQ_API_KEY")

        if not groq_key:
            message = "Please enter your Groq API key."
        elif not question:
            message = "Please enter a question."
        else:
            try:
                from searche import search  # local file: searche.py
                response = search(question, groq_key)
                session["chat_history"].append({"role": "user", "content": question})
                session["chat_history"].append({"role": "assistant", "content": response})
                session.modified = True
                save_user_chat_history(session["user_id"], session["chat_history"])
            except Exception as e:
                message = f"Search error: {str(e)}"

    return render_template("search.html", message=message, response=response)


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    message = ""
    df = None
    df_html = ""  
    df_preview_html = ""
    result_html = ""
    code_generated = ""
    cols = [] 
    current_file_name = session.get("uploaded_file_name")
    selected_target_variable = session.get("selected_target_variable", "")
    selected_chart_type = session.get("selected_chart_type", "auto")

   
    if 'chat_history' not in session:
        session['chat_history'] = load_user_chat_history(session["user_id"])

    # If a file is already loaded for this session, read it so columns/preview are available
    if session.get("uploaded_file_path"):
        df, cols, df_html, err = _load_df_from_path(session.get("uploaded_file_path"))
        if not err and df is not None:
            df_preview_html = df.head().to_html(classes="table table-auto w-full", index=False)
        elif err:
            # Don't block the page; just show the error and let user re-upload
            message = err

    if request.method == "POST":
        file = request.files.get("file")
        query = request.form.get("query", "").strip()  # Strip whitespace
        chart_type = (request.form.get("chart_type") or "auto").strip().lower()
        selected_target_variable = (request.form.get("target_variable") or "").strip()
        selected_chart_type = chart_type or "auto"
        session["selected_target_variable"] = selected_target_variable
        session["selected_chart_type"] = selected_chart_type
        session.modified = True

        # If a new file was uploaded, save it and mark it as current for the session.
        if file and file.filename:
            old_path = session.get("uploaded_file_path")
            saved_path, save_err = _save_uploaded_file(file)
            if save_err:
                message = save_err
            else:
                if old_path and old_path != saved_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass
                session["uploaded_file_path"] = saved_path
                session["uploaded_file_name"] = os.path.basename(saved_path)
                current_file_name = session["uploaded_file_name"]

        # Load dataframe from the current saved file (supports multi-query without re-upload).
        df, cols, df_html, err = _load_df_from_path(session.get("uploaded_file_path"))
        if err:
            message = err or "Please upload a file."
        else:
            df_preview_html = df.head().to_html(classes="table table-auto w-full", index=False)

            if selected_target_variable and selected_target_variable not in cols:
                selected_target_variable = ""
                session["selected_target_variable"] = ""
                session.modified = True

            if query:
                # Only require Groq key when running a query
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key:
                    message = "Please set your Groq API key to run queries."
                else:
                    try:
                        prompt = f"""
You are a Python data analyst. Given a pandas DataFrame named `df` with columns: {list(cols)},

Write **only** the Python code (no explanations, no markdown) to answer this question:

Question: {query}

- You can use: `pandas`, `numpy`, `matplotlib`, `seaborn`, and the helper module `analysis_tools`.
- Prefer calling `analysis_tools` for advanced analysis like logistic regression, linear regression, clustering, and statistical tests.
- The user-selected chart type is: {selected_chart_type}. If it's not "auto", create that type of chart unless impossible.
- Support common charts: bar, line, scatter, histogram, box, violin, heatmap, pie, area.
- The user-selected target variable (column) is: {selected_target_variable or "none"}.
- If a target variable is provided and the question is ambiguous, assume the question is about the target variable.
- When creating a plot and a target variable is provided, prefer using it as the y-variable (or the main numeric distribution) unless the user explicitly asks otherwise.
- Store the final *data* answer (like a number, string, or DataFrame) in a variable named `result`.
- If plotting, just create the plot (e.g., `sns.histplot(df['column'])`). **DO NOT** save the plot to a file or assign the plot object to the `result` variable. The server will save the plot automatically.
"""

                        client = Groq(api_key=groq_key)
                        chat_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile", # Note: Check if this model name is correct/available
                            temperature=0.2,
                            max_tokens=1024
                        )
                        raw_response = chat_completion.choices[0].message.content.strip()

                        # Clean code block
                        code_generated = raw_response
                        if code_generated.startswith("```python"):
                            code_generated = code_generated[10:]
                        if code_generated.startswith("```"):
                            code_generated = code_generated[3:]
                        if code_generated.endswith("```"):
                            code_generated = code_generated[:-3]
                        code_generated = code_generated.strip()

                        # Save to session
                        session['chat_history'].append({"role": "user", "content": query}) # Save user query
                        session['chat_history'].append({"role": "assistant", "content": code_generated})
                        session.modified = True
                        save_user_chat_history(session["user_id"], session["chat_history"])

                        # Safe execution environment
                        local_vars = {
                            "df": df.copy(),  # Prevent modification of original
                            "pd": pd,
                            "plt": plt,
                            "sns": sns,
                            "np": np,
                            "io": io,
                            "analysis_tools": analysis_tools,
                        }

                        # Clear any previous plot from matplotlib's global state
                        plt.clf()

                        # Execute generated code
                        exec(code_generated, {}, local_vars)

                        result = local_vars.get("result")
                        plot_path = os.path.join(app.root_path, "static", "result_plot.png")
                        # Add a cache-busting query param to the URL
                        plot_url = f"/static/result_plot.png?v={os.times().system}"

                        # Clear old plot file if it exists
                        if os.path.exists(plot_path):
                            os.remove(plot_path)

                        # Handle the 'result' variable (data)
                        if result is not None:
                            if isinstance(result, pd.DataFrame):
                                result_html = result.to_html(classes="table table-auto w-full", index=False)
                            elif isinstance(result, (pd.Series, list, dict, str, int, float)):
                                result_html = f'<pre class="bg-gray-100 p-2 rounded">{str(result)}</pre>'
                            else:
                                result_html = "Result generated (type not directly displayable)."
                        else:
                            # Don't say "no result" if a plot was the intended output
                            if not plt.gcf().get_axes():
                                result_html = "Code executed but no `result` variable or plot was found."

                        # Handle plot generation
                        # Check if the code *actually* created a plot by seeing if there are axes
                        if 'plt' in local_vars and plt.gcf().get_axes():
                            plt.tight_layout()
                            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                            plt.close() # Close figure to free memory
                            # Add image tag to the result
                            result_html += f'<br><img src="{plot_url}" alt="Generated Plot" class="mt-4 rounded border">'
                        
                        # Clear figure again just in case
                        plt.clf()
                        plt.close()

                    except Exception as e:
                        message = f"Error executing code: {str(e)}"
                        import traceback
                        print(traceback.format_exc())
            elif file and file.filename:
                message = "File loaded successfully. You can select a target column and run queries."

    # In app.py, change the return statement:
    return render_template(
        "index.html",
        message=message,
        df_html=df_html,
        df_preview_html=df_preview_html,
        column_names=cols,             # Changed from cols=cols to match HTML
        summary_html=df.describe().to_html(classes="table") if df is not None else None, # Added
        code_generated=code_generated,
        result_html=result_html,
        chat_history=session['chat_history'],
        username=session.get("username"),
        current_file_name=current_file_name,
        selected_target_variable=selected_target_variable,
        selected_chart_type=selected_chart_type
)


if __name__ == "__main__":
    # debug=True allows the server to auto-reload when you save changes
    app.run(debug=True, port=5000)

    