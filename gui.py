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
        # Queue for receiving prints/status from the bot
        self.msg_queue = queue.Queue()
        # Event to pause/unpause the bot during written surveys
        self.survey_event = threading.Event()
        # Shared dictionary to pass the survey answer back to the bot
        self.survey_data = {"answer": ""}

        # --- UI LAYOUT ---

        # Header / Status Frame
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.pack(pady=10, padx=20, fill="x")

        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray"
        )
        self.status_label.pack(pady=10)

        # Chat Log (Scrollable Text Box)
        self.chat_textbox = ctk.CTkTextbox(self, state="disabled", font=ctk.CTkFont(size=14))
        self.chat_textbox.pack(pady=10, padx=20, fill="both", expand=True)

        # Control Panel
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=10, padx=20, fill="x")

        self.start_button = ctk.CTkButton(self.control_frame, text="Start Session", command=self.start_experiment)
        self.start_button.pack(side="left", padx=10, pady=10)

        # Survey Input Box (Hidden initially)
        self.survey_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Type your answer here...", width=400)
        self.survey_submit = ctk.CTkButton(self.control_frame, text="Submit", command=self.submit_survey)

        # Start the queue polling loop (Runs every 100ms)
        self.after(100, self.poll_queue)

    # --- UI UPDATE METHODS ---

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
                gui_input_dict=self.survey_data
            )
        except Exception as e:
            self.msg_queue.put({"type": "status", "text": f"Error: {str(e)}", "color": "red"})
        finally:
            # Always re-enable the start button when finished (using safe after callback)
            self.after(0, lambda: self.start_button.configure(state="normal"))


if __name__ == "__main__":
    app = ResearchDemoUI()
    app.mainloop()