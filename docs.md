Documentation

There are 3 main directories which are core to this project:
- human_anatomy
- nigerian_entertainment
- nigerian_history

These directories are the main topics in which questions will be generated. Each folder will have multiple sub folders and those sub folders will have sub folders of their on going over a depth of 3 and more as time passes.

### CORE SCRIPTS
- delete_all_pj: This script deletes all previous question files within the directory. I created this when I need som sort of reset mechanism

- generate_questions.py: This is the main script for generating questions within the directory. This is the money maker

- refresh_db.py: This looks at all the questions in the csv, which should reflect all the questions in the db and performs the following:
    - checks if theres any missing question thats in the csv and not in the db and vice versa
    - deduplicates such questions making sure similar questions don't enter