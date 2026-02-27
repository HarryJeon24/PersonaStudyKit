import customtkinter as ctk
import threading
import queue
import bot_core

# Set up the visual theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ResearchDemoUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Voice Assistant Research Demo")
        self.geometry("800x600")

        # --- THREADING COMMUNICATION PIPES ---
        self.msg_queue = queue.Queue()
        self.survey_event = threading.Event()
        self.survey_data = {"answer": ""}

        # User details
        self.username = ""

        # --- UI LAYOUTS ---
        self.build_login_frame()
        self.build_main_frame()

        # Start the queue polling loop (Runs every 100ms)
        self.after(100, self.poll_queue)

    def build_login_frame(self):
        """Creates the initial login and sign up screen."""
        self.login_frame = ctk.CTkFrame(self)
        self.login_frame.pack(pady=50, padx=50, fill="both", expand=True)

        self.login_title = ctk.CTkLabel(
            self.login_frame,
            text="Research Participant Portal",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.login_title.pack(pady=(50, 20))

        self.username_entry = ctk.CTkEntry(
            self.login_frame,
            placeholder_text="Enter Username",
            width=300,
            font=ctk.CTkFont(size=14)
        )
        self.username_entry.pack(pady=10)

        self.password_entry = ctk.CTkEntry(
            self.login_frame,
            placeholder_text="Enter Password",
            show="*",
            width=300,
            font=ctk.CTkFont(size=14)
        )
        self.password_entry.pack(pady=10)

        # Label to display error messages (e.g., "Invalid Password")
        self.auth_error_label = ctk.CTkLabel(self.login_frame, text="", text_color="red")
        self.auth_error_label.pack(pady=5)

        # Frame to hold the side-by-side buttons
        btn_frame = ctk.CTkFrame(self.login_frame, fg_color="transparent")
        btn_frame.pack(pady=20)

        self.login_btn = ctk.CTkButton(
            btn_frame,
            text="Login",
            command=self.do_login,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.login_btn.pack(side="left", padx=10)

        self.signup_btn = ctk.CTkButton(
            btn_frame,
            text="Sign Up",
            command=self.do_signup,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="gray30",
            hover_color="gray40"
        )
        self.signup_btn.pack(side="left", padx=10)

    def build_main_frame(self):
        """Creates the main experiment panel (hidden initially)."""
        self.main_frame = ctk.CTkFrame(self)

        # Header / Status Frame
        self.header_frame = ctk.CTkFrame(self.main_frame)
        self.header_frame.pack(pady=10, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray"
        )
        self.status_label.pack(side="left", padx=20, pady=10)

        # Logout Button
        self.logout_btn = ctk.CTkButton(
            self.header_frame,
            text="Log Out",
            command=self.do_logout,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "#DCE4EE")
        )
        self.logout_btn.pack(side="right", padx=20, pady=10)

        # Chat Log (Scrollable Text Box)
        self.chat_textbox = ctk.CTkTextbox(self.main_frame, state="disabled", font=ctk.CTkFont(size=14))
        self.chat_textbox.pack(pady=10, padx=20, fill="both", expand=True)

        # Control Panel
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(pady=10, padx=20, fill="x")

        self.start_button = ctk.CTkButton(self.control_frame, text="Start Session", command=self.start_experiment)
        self.start_button.pack(side="left", padx=10, pady=10)

        # Survey Input Box (Hidden initially)
        self.survey_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Type your answer here...", width=400)
        self.survey_submit = ctk.CTkButton(self.control_frame, text="Submit", command=self.submit_survey)

    # --- UI UPDATE METHODS ---

    def do_login(self):
        """Validates username/password and swaps frames if successful."""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        success, msg = bot_core.authenticate_user(username, password)

        if success:
            self.username = username
            self.login_frame.pack_forget()
            self.main_frame.pack(fill="both", expand=True)
            self.update_status(f"Logged in as '{self.username}'", color="white")

            # Clear fields for security upon returning
            self.password_entry.delete(0, 'end')
            self.auth_error_label.configure(text="")
        else:
            self.auth_error_label.configure(text=msg, text_color="red")

    def do_signup(self):
        """Attempts to register a new user."""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        success, msg = bot_core.register_user(username, password)

        if success:
            self.auth_error_label.configure(text="Sign up successful! Please log in.", text_color="green")
        else:
            self.auth_error_label.configure(text=msg, text_color="red")

    def do_logout(self):
        """Returns to the login screen."""
        self.username = ""
        self.username_entry.delete(0, 'end')
        self.password_entry.delete(0, 'end')
        self.main_frame.pack_forget()
        self.login_frame.pack(pady=50, padx=50, fill="both", expand=True)
        # Clear the chat textbox for the next user
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.delete("0.0", "end")
        self.chat_textbox.configure(state="disabled")

    def append_to_chat(self, speaker, text, color="white"):
        """Safely inserts text into the chat log."""
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.insert("end", f"{speaker}: ", "bold")
        self.chat_textbox.insert("end", f"{text}\n\n")
        self.chat_textbox.configure(state="disabled")
        self.chat_textbox.yview("end")  # Auto-scroll to bottom

    def update_status(self, text, color="white"):
        """Safely updates the top status indicator."""
        self.status_label.configure(text=f"Status: {text}", text_color=color)

    def poll_queue(self):
        """Checks the queue every 100ms for updates from the bot_core thread."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()

                if msg["type"] == "chat":
                    self.append_to_chat(msg["speaker"], msg["text"])
                elif msg["type"] == "status":
                    self.update_status(msg["text"], msg.get("color", "white"))
                elif msg["type"] == "survey_prompt":
                    # Show the survey input box
                    self.survey_entry.pack(side="left", padx=10, pady=10)
                    self.survey_submit.pack(side="left", padx=10, pady=10)
                    self.survey_entry.focus()  # Auto-focus the cursor in the box
                elif msg["type"] == "hide_survey":
                    # Hide the survey input box
                    self.survey_entry.pack_forget()
                    self.survey_submit.pack_forget()

        except queue.Empty:
            pass

        # Keep polling forever
        self.after(100, self.poll_queue)

    # --- ACTION METHODS ---

    def start_experiment(self):
        """Disables the start button and launches the core bot logic in a background thread."""
        self.start_button.configure(state="disabled")
        self.logout_btn.configure(state="disabled")  # Prevent logging out mid-session

        self.chat_textbox.configure(state="normal")
        self.chat_textbox.delete("0.0", "end")  # Clear previous logs
        self.chat_textbox.configure(state="disabled")

        # Launch the bot_core in a separate thread so it doesn't freeze the UI
        bot_thread = threading.Thread(target=self.run_bot_logic, daemon=True)
        bot_thread.start()

    def submit_survey(self):
        """Captures the entered text and unpauses the bot_core thread."""
        answer = self.survey_entry.get().strip()
        if not answer:
            return  # Ignore empty submits

        # 1. Save the answer to the shared dictionary
        self.survey_data["answer"] = answer

        # 2. Clear the UI entry box
        self.survey_entry.delete(0, 'end')

        # 3. Trigger the event to unblock the bot_core thread
        self.survey_event.set()

    # --- BACKGROUND WORKER LOGIC ---

    def run_bot_logic(self):
        """THIS RUNS IN THE BACKGROUND THREAD."""
        # Ensure the event flag is cleared before starting
        self.survey_event.clear()

        try:
            # Pass our communication pipes directly into your core code!
            bot_core.run_experiment(
                gui_queue=self.msg_queue,
                gui_input_event=self.survey_event,
                gui_input_dict=self.survey_data,
                username=self.username
            )
        except Exception as e:
            self.msg_queue.put({"type": "status", "text": f"Error: {str(e)}", "color": "red"})
        finally:
            # Always re-enable buttons when finished
            self.after(0, lambda: self.start_button.configure(state="normal"))
            self.after(0, lambda: self.logout_btn.configure(state="normal"))


if __name__ == "__main__":
    app = ResearchDemoUI()
    app.mainloop()