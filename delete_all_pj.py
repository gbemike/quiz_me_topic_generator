import os

def delete_previous_questions_files(start_path="."):
    deleted_files = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            if filename == "previous_questions.json":
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
    print(f"\n✅ Done. Deleted {deleted_files} 'previous_questions.json' file(s).")

if __name__ == "__main__":
    delete_previous_questions_files()
