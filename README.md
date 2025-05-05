# text-to-sql

Text-to-SQL is a natural language processing (NLP) task that involves translating a user's question or command, written in plain language, into a SQL query that can be executed on a database. This task enables users to interact with databases without needing to know SQL syntax, making data access more intuitive and user-friendly. This task is not simple, however. It requires understanding both the intent behind the question and the underlying database schema to generate accurate and executable queries. Text-to-SQL is commonly used in applications like data analytics tools, chatbots, and business intelligence systems, where users seek to retrieve or analyze data using natural language. 

## Resources

Included are the notebook and JSON files used with the Text-to-SQL project. To use these notebooks yourself, install these dependencies:

- `transformers`
- `datasets`
- `evaluate`
- `torch`
- `anthropic`
- `seaborn`
- `pandas`

Notebooks:

- `model-finetune-wikiSQL` - a resource to finetune different transformers available through HuggingFace for comparative analysis with the Text-to-SQL task.
- `model-test-wikiSQL` - used to perform comparative analysis between models previously finetuned on WikiSQL dataset.
- `examine-sql-create-context` - used to verify some basic quality checks for the SQL-Create-Context dataset
- `CodeT5-base-sql-create-context` - used to finetune `CodeT5` on the SQL-Create-Context dataset for use with the project's app.
- `claude-test-sql-create-context` - a testing resource to compare between `CodeT5` and `Claude` for Text-to-SQL under a new prompting method.

For completeness, JSON files produced by the project are also included. These primarily served to supplement notebook files during analysis.

If you would like to try out `app.py`, you only need to run the `app_dependencies_and_download_model.py` script. This will verify or install `transformers`, `anthropic`, and `torch` as well as download the project's version of `CodeT5` from HuggingFace. The model is stored locally, taking about 900MB of space. 

## The App

After running the above script, you can try out the application. `app.py` is a Text-to-SQL utility, which uses `tkinter` to provide a GUI. It uses two separate models: the locally stored `CodeT5` that was finetuned for the task, and `Claude 3.7 Sonnet` via the Anthropic API. To summarize, when provided with a required Anthropic API key and database, table schema may be requested and user questions may be provided. The interface then progresses to a step where models are prompted and the returned queries are provided back. From here, you have the option to use the query from `CodeT5`, `Claude`, or even enter your own. Returned results are also able to be stored away for later use as JSON format files.

Keep in mind that this is only a work in progress, and you may run into errors. Also, The local tranformer model is large, and will be loaded into memory when the app is run. Be sure and provide it some time to load. 
