import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sqlite3
import json
import threading
import anthropic
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # keep tensorflow quiet except errors
CLAUDE = "claude-3-7-sonnet-20250219"
PATH = 'finetuned/codet5-base-wikisql'
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(PATH)
model.to(device)


def get_database_schema(db_path):
    """Extract and return the database schema as a formatted string."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_text = ""
    
    for table in tables:
        table_name = table[0]
        # Skip sqlite system tables
        if table_name.startswith('sqlite_'):
            continue
            
        schema_text += f"Table: {table_name}\n"
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for col in columns:
            col_id, name, type_name, not_null, default_val, pk = col
            constraints = []
            if pk:
                constraints.append("PRIMARY KEY")
            if not_null:
                constraints.append("NOT NULL")
            if default_val is not None:
                constraints.append(f"DEFAULT {default_val}")
                
            constraint_str = ", ".join(constraints)
            if constraint_str:
                schema_text += f"  - {name} ({type_name}, {constraint_str})\n"
            else:
                schema_text += f"  - {name} ({type_name})\n"
        
        schema_text += "\n"
    
    conn.close()
    return schema_text
    

def get_table_schema(db_path, table_name):
    """construct a schema string to use for prompt context"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    result = f"CREATE TABLE {table_name} (\n" + "\n".join([f"{col[1]} {col[2]}" for col in schema]) + "\n)"
    
    conn.close()
    return result



def ask_claude(question, schema, api_client):
    """function to interact with anthropic api"""
    prompt = f"""Here is the schema for a database:
{schema}
Given this schema, can you output a SQL query to answer the following question? 
Only output the SQL query, use double quotes instead of single quotes in the query, and no markdown formatting or newline characters.
Question: {question}
"""
    try:
        response = api_client.messages.create(
            model=CLAUDE,
            max_tokens=512,
            messages=[{
                "role": 'user', "content":  prompt
            }]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error querying API: {e}"




def ask_codet5(question, schema):
    """function to generate from finetuned CodeT5-base"""
    prompt = 'schema: \n' + str(schema)[:420] + '\n\ntranslate to SQL: ' + str(question)
    inputs = tokenizer(prompt, truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=256)[0]
    prediction = tokenizer.decode(output, skip_special_tokens=True)
            
    return prediction

    
def ask(api_client, table_name, question, db_path):
    """Send the user's question to the API and return responses from Claude and CodeT5."""
    # Get table schema for context
    schema = get_table_schema(db_path, table_name)
    
    # Have models make predictions
    claude_response = ask_claude(question, schema, api_client)
    codet5_response = ask_codet5(question, schema)
    
    return claude_response, codet5_response


def extract_sql_query(response_text):
    """Extract SQL query from a text response."""
    import re
    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.search(sql_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches.group(1).strip()
    
    code_pattern = r"```\s*(.*?)\s*```"
    matches = re.search(code_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches.group(1).strip()
    
    # Fallback: assume entire response is the SQL query
    return response_text.strip()


def execute_query(db_path, query):
    """Execute an SQL query on the database and return results."""
    conn = sqlite3.connect(db_path)
    # Use Row as row_factory to get dictionaries
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(query)
    
    # Convert sqlite3.Row objects to dictionaries
    results = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return results


def format_query_results(results, max_rows=10):
    """Format query results as a string, limiting to max_rows."""
    if not results:
        return "No results found."
    
    # Limit number of rows
    results_to_show = results[:max_rows]
    
    # Get column names from the first row
    columns = list(results_to_show[0].keys())
    
    # Calculate column widths
    col_widths = {col: len(col) for col in columns}
    for row in results_to_show:
        for col in columns:
            col_widths[col] = max(col_widths[col], len(str(row[col] or "")))
    
    # Create header
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    separator = "-" * len(header)
    
    # Create rows
    rows = []
    for row in results_to_show:
        formatted_row = " | ".join(str(row[col] or "").ljust(col_widths[col]) for col in columns)
        rows.append(formatted_row)
    
    # Combine all parts
    return f"{header}\n{separator}\n" + "\n".join(rows)

############################################################
# Start of class
###########################################################

class DatabaseQueryApp:
    """Main application class for the database query GUI."""
    
    def __init__(self, root):
        """Initialize the application with the root window."""
        self.root = root
        self.root.title("Text-to-SQL")
        self.root.geometry("900x700")
        
        # Application state variables
        self.api_key = None
        self.db_path = None
        self.anthropic_client = None
        self.selected_table = None
        self.current_question = None
        self.claude_response = None
        self.codet5_response = None
        self.query_results = None
        self.selected_query_option = tk.StringVar(value="claude")
        
        # Create the frames
        self.frame1 = ttk.Frame(self.root, padding="20")
        self.frame2 = ttk.Frame(self.root, padding="20")
        
        # Start with Frame 1
        self.setup_frame1()
        self.frame1.pack(fill=tk.BOTH, expand=True)
    
    def setup_frame1(self):
        """Set up the first frame for API key and database selection."""
        # Description
        description_label = ttk.Label(self.frame1, text=
"""Welcome to the Text-to-SQL GUI, where you can get help accessing your database by asking natural questions. 
Provide your API key from Anthropic and browse for your database file below to get started.""", font=("Arial", 12))
        description_label.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky="w")

        #Configure Heading
        title_label = ttk.Label(self.frame1, text="Setup Configuration", font=("Arial", 16, "bold"))
        title_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky="w")
        
        # API Key
        api_key_label = ttk.Label(self.frame1, text="Anthropic API Key:")
        api_key_label.grid(row=2, column=0, sticky="w", pady=5)
        
        self.api_key_entry = ttk.Entry(self.frame1, width=50, show="*")
        self.api_key_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        
        # Database Selection
        db_path_label = ttk.Label(self.frame1, text="SQLite Database Path:")
        db_path_label.grid(row=3, column=0, sticky="w", pady=5)
        
        self.db_path_var = tk.StringVar()
        db_path_entry = ttk.Entry(self.frame1, textvariable=self.db_path_var, width=50, state="readonly")
        db_path_entry.grid(row=3, column=1, sticky="ew", pady=5, padx=5)
        
        browse_button = ttk.Button(self.frame1, text="Browse...", command=self.browse_db)
        browse_button.grid(row=3, column=2, pady=5, padx=5)
        
        # Continue Button
        continue_button = ttk.Button(self.frame1, text="Continue", command=self.validate_and_continue)
        continue_button.grid(row=4, column=1, pady=20)
        
        # Configure grid weights
        self.frame1.columnconfigure(1, weight=1)

    def validate_and_continue(self):
        """Validate API key and database path before proceeding to Frame 2."""
        api_key = self.api_key_entry.get().strip()
        db_path = self.db_path_var.get()
        
        if not api_key:
            messagebox.showerror("Validation Error", "Please enter an API key.")
            return
        
        if not db_path:
            messagebox.showerror("Validation Error", "Please select a database file.")
            return
        
        # Validate database connection
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to connect to database: {str(e)}")
            return
        
        # Validate API key
        try:
            client = anthropic.Anthropic(api_key=api_key)
            # Just a simple test to see if the API key format is valid
            # We don't actually send a request to avoid unnecessary API usage
            if not api_key.startswith(("sk-ant-", "sk-")):
                messagebox.showerror("API Key Error", "API key format appears invalid.")
                return
        except Exception as e:
            messagebox.showerror("API Error", f"Failed to initialize API client: {str(e)}")
            return
        
        # Store valid values
        self.api_key = api_key
        self.db_path = db_path
        self.anthropic_client = client

        # Switch to frame 2
        self.frame1.pack_forget()
        self.setup_frame2()
        
        # Log successful setup
        self.log_message("Setup complete. Database connection and API key validated successfully.")
        self.log_message(f"Database path: {db_path}")
        self.log_message("Please select a table to begin.")
    
    def toggle_table_selection(self):
        """Toggle the visibility of table selection options based on user choice."""
        if self.know_table.get() == "no":
            self.show_schema_button.grid()
        else:
            self.show_schema_button.grid_remove()
    
    def setup_frame2(self):
        """Set up the second frame for the multi-step database query process."""
        # Output Text Area
        output_frame = ttk.LabelFrame(self.frame2, text="Output Log")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
        
        # Create frames for each step
        self.step1_frame = ttk.LabelFrame(self.frame2, text="Step 1: Table Selection")
        self.step2_frame = ttk.LabelFrame(self.frame2, text="Step 2: Ask a Question")
        self.step3_frame = ttk.LabelFrame(self.frame2, text="Step 3: Choose Query to Execute")
        self.step4_frame = ttk.LabelFrame(self.frame2, text="Step 4: Post-Execution Options")
        
        # Setup Step 1 content
        self.setup_step1()
        self.step1_frame.pack(fill=tk.X, pady=5)

        self.frame2.pack(fill=tk.BOTH, expand=True)
    
    def setup_step1(self):
        """Set up the first step frame for table selection."""
        # Table name label and entry
        table_name_label = ttk.Label(self.step1_frame, text="Table Name:")
        table_name_label.grid(row=0, column=0, sticky="w", pady=5)
    
        self.table_name_entry = ttk.Entry(self.step1_frame, width=30)
        self.table_name_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)
    
        # Show schema button
        self.show_schema_button = ttk.Button(self.step1_frame, text="Show Database Schema",
                                             command=self.show_schema)
        self.show_schema_button.grid(row=1, column=0, columnspan=2, pady=10)
    
        # Continue button
        continue_button = ttk.Button(self.step1_frame, text="Continue to Question",
                                     command=self.proceed_to_step2)
        continue_button.grid(row=1, column=2, pady=10)
    
    def setup_step2(self):
        """Set up the second step frame for asking a question."""
        # Question entry
        question_label = ttk.Label(self.step2_frame, text="Enter your question about the data:")
        question_label.grid(row=0, column=0, sticky="w", pady=5)
        
        self.question_entry = ttk.Entry(self.step2_frame, width=70)
        self.question_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Submit button
        submit_button = ttk.Button(self.step2_frame, text="Submit Question", command=self.submit_question)
        submit_button.grid(row=2, column=0, pady=10)
        
        # Back button
        back_button = ttk.Button(self.step2_frame, text="Back to Table Selection", command=self.back_to_step1)
        back_button.grid(row=2, column=1, pady=10)
    
    def setup_step3(self):
        """Set up the third step frame for query selection and execution."""
        # Query selection
        query_label = ttk.Label(self.step3_frame, text="Choose which query to execute:")
        query_label.grid(row=0, column=0, sticky="w", pady=5, columnspan=2)
        
        claude_radio = ttk.Radiobutton(self.step3_frame, text="Use Claude's Query", 
                                     variable=self.selected_query_option, value="claude",
                                     command=self.toggle_custom_query)
        codet5_radio = ttk.Radiobutton(self.step3_frame, text="Use CodeT5's Query", 
                                     variable=self.selected_query_option, value="codet5",
                                     command=self.toggle_custom_query)
        custom_radio = ttk.Radiobutton(self.step3_frame, text="Enter Custom Query", 
                                     variable=self.selected_query_option, value="custom",
                                     command=self.toggle_custom_query)
        back_radio = ttk.Radiobutton(self.step3_frame, text="Go Back to Question", 
                                   variable=self.selected_query_option, value="back",
                                   command=self.toggle_custom_query)
        
        claude_radio.grid(row=1, column=0, sticky="w", padx=10)
        codet5_radio.grid(row=1, column=1, sticky="w", padx=10)
        custom_radio.grid(row=2, column=0, sticky="w", padx=10)
        back_radio.grid(row=2, column=1, sticky="w", padx=10)
        
        # Custom query text area
        self.custom_query_frame = ttk.Frame(self.step3_frame)
        self.custom_query_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        custom_query_label = ttk.Label(self.custom_query_frame, text="Enter custom SQL query:")
        custom_query_label.pack(anchor="w", pady=(5, 0))
        
        self.custom_query_text = scrolledtext.ScrolledText(self.custom_query_frame, height=4, width=60)
        self.custom_query_text.pack(fill=tk.X, pady=5)
        
        # Initially hide custom query text area
        self.custom_query_frame.grid_remove()
        
        # Execute button
        execute_button = ttk.Button(self.step3_frame, text="Execute Query", command=self.execute_selected_query)
        execute_button.grid(row=4, column=0, columnspan=2, pady=10)
    
    def setup_step4(self):
        """Set up the fourth step frame for post-execution options."""
        # Save to JSON button
        save_button = ttk.Button(self.step4_frame, text="Save Results to JSON", command=self.save_to_json)
        save_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Restart button
        restart_button = ttk.Button(self.step4_frame, text="Start New Query", command=self.restart_query)
        restart_button.grid(row=0, column=1, padx=10, pady=10)
    
    def browse_db(self):
        """Open a file dialog to browse for a SQLite database file."""
        db_path = filedialog.askopenfilename(
            title="Select SQLite Database",
            filetypes=[("SQLite Database", "*.db *.sqlite *.sqlite3"), ("All Files", "*.*")]
        )
        
        if db_path:
            self.db_path_var.set(db_path)
    
    def show_schema(self):
        """Show the database schema in the output text area."""
        try:
            schema = get_database_schema(self.db_path)
            self.log_message(f"Database Schema: \n{schema}")

        except Exception as e:
            self.log_message(f"Error retrieving schema: {str(e)}")
    
    def proceed_to_step2(self):
        """Validate table name and proceed to step 2."""
        table_name = self.table_name_entry.get().strip()
        
        if not table_name:
            messagebox.showerror("Validation Error", "Please enter a table name.")
            return
        
        # Verify the table exists
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cursor.fetchone():
                messagebox.showerror("Table Error", f"Table '{table_name}' does not exist in the database.")
                conn.close()
                return
            conn.close()
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to verify table: {str(e)}")
            return
        
        # Store the selected table
        self.selected_table = table_name
        self.log_message(f"Selected table: {table_name}")
        
        # Setup and show step 2
        self.setup_step2()
        self.step1_frame.pack_forget()
        self.step2_frame.pack(fill=tk.X, pady=5)
    
    def back_to_step1(self):
        """Go back to step 1."""
        self.step2_frame.pack_forget()
        self.step1_frame.pack(fill=tk.X, pady=5)
    
    def submit_question(self):
        """Submit the question and get model responses."""
        question = self.question_entry.get().strip()
        
        if not question:
            messagebox.showerror("Validation Error", "Please enter a question.")
            return
        
        self.current_question = question
        self.log_message(f"Question: {question}")
        
        # Disable the submit button during processing
        for widget in self.step2_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.DISABLED)
        
        # Use a thread to avoid freezing the UI
        threading.Thread(target=self.process_question, daemon=True).start()
    
    def process_question(self):
        """Process the question in a separate thread."""
        try:
            # This would actually call an API, but we'll simulate it for demonstration
            self.log_message("Processing question... This may take a moment.")
            
            # Simulate API call
            claude_response, codet5_response = ask(
                self.anthropic_client, 
                self.selected_table, 
                self.current_question,
                self.db_path
            )
            
            self.claude_response = claude_response
            self.codet5_response = codet5_response
            
            # Update UI on the main thread
            self.root.after(0, self.show_model_responses)
        except Exception as e:
            self.root.after(0, lambda: self.log_message(f"Error processing question: {str(e)}"))
            # Re-enable buttons
            self.root.after(0, self.enable_step2_buttons)
    
    def show_model_responses(self):
        """Show model responses in the output area."""
        self.log_message("\nClaude's Response:")
        self.log_message(self.claude_response)
        
        self.log_message("\nCodeT5's Response:")
        self.log_message(self.codet5_response)
        
        # Enable buttons
        self.enable_step2_buttons()
        
        # Set up and show step 3
        self.setup_step3()
        self.step2_frame.pack_forget()
        self.step3_frame.pack(fill=tk.X, pady=5)
    
    def enable_step2_buttons(self):
        """Re-enable buttons in step 2."""
        for widget in self.step2_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.NORMAL)
    
    def toggle_custom_query(self):
        """Show or hide the custom query text area based on selection."""
        if self.selected_query_option.get() == "custom":
            self.custom_query_frame.grid()
        else:
            self.custom_query_frame.grid_remove()
    
    def execute_selected_query(self):
        """Execute the selected query."""
        option = self.selected_query_option.get()
        
        if option == "back":
            self.step3_frame.pack_forget()
            self.step2_frame.pack(fill=tk.X, pady=5)
            return
        
        # Get the appropriate query
        query = ""
        if option == "claude":
            # Extract SQL query from Claude's response
            query = extract_sql_query(self.claude_response)
        elif option == "codet5":
            # Extract SQL query from CodeT5's response
            query = extract_sql_query(self.codet5_response)
        elif option == "custom":
            query = self.custom_query_text.get("1.0", tk.END).strip()
        
        if not query:
            messagebox.showerror("Query Error", "Failed to extract or get a valid SQL query.")
            return
        
        self.log_message(f"\nExecuting query:\n{query}")
        
        try:
            # Execute the query
            results = execute_query(self.db_path, query)
            self.query_results = results
            
            # Display results
            if isinstance(results, list) and results:
                result_str = format_query_results(results, max_rows=10)
                self.log_message("\nQuery Results:")
                self.log_message(result_str)
                
                if len(results) > 10:
                    self.log_message(f"(Showing 10 of {len(results)} rows)")
            else:
                self.log_message("\nQuery executed successfully with no results to display.")
            
            # Setup and show step 4
            self.setup_step4()
            self.step3_frame.pack_forget()
            self.step4_frame.pack(fill=tk.X, pady=5)
        except Exception as e:
            self.log_message(f"Error executing query: {str(e)}")
    
    def save_to_json(self):
        """Save query results to a JSON file."""
        if not self.query_results:
            messagebox.showinfo("No Results", "There are no results to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.query_results, f, indent=2, ensure_ascii=False)
                self.log_message(f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")
    
    def restart_query(self):
        """Restart the query process from step 1."""
        # Reset state variables
        self.selected_table = None
        self.current_question = None
        self.claude_response = None
        self.codet5_response = None
        self.query_results = None
        self.selected_query_option.set("claude")
        
        # Clear input fields
        self.table_name_entry.delete(0, tk.END)
        try:
            self.question_entry.delete(0, tk.END)
        except:
            pass  # Step 2 might not be created yet
        
        try:
            self.custom_query_text.delete("1.0", tk.END)
        except:
            pass  # Step 3 might not be created yet
        
        # Hide all step frames
        for frame in [self.step1_frame, self.step2_frame, self.step3_frame, self.step4_frame]:
            try:
                frame.pack_forget()
            except:
                pass  # Some frames might not exist yet
        
        # Show step 1
        self.step1_frame.pack(fill=tk.X, pady=5)
        
        self.log_message("\n--- Starting new query ---")
    
    def log_message(self, message):
        """Log a message to the output text area."""
        try:
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, message + "\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        except:
            # Output text may not exist yet
            pass

############################################################
# End of class
###########################################################


if __name__ == "__main__":
    root = tk.Tk()
    app = DatabaseQueryApp(root)
    root.mainloop()