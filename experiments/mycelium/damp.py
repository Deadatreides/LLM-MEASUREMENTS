import os

# Настройки: какие расширения собирать и что игнорировать
EXTENSIONS = ('.py', '.js', '.ts', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.php', '.html', '.css', '.json', '.yaml', '.yml', '.txt', '.conf', '.ini', '.md')
IGNORE_DIRS = {'__pycache__', '.git', 'node_modules', 'venv', '.venv', 'dist', 'build'}

def create_dump(root_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as dump:
        for root, dirs, files in os.walk(root_path):
            # Пропускаем ненужные папки
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file.endswith(EXTENSIONS):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, root_path)
                    
                    dump.write(f"\n{'='*80}\n")
                    dump.write(f"FILE: {relative_path}\n")
                    dump.write(f"{'='*80}\n\n")
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            dump.write(f.read())
                    except Exception as e:
                        dump.write(f"[Ошибка чтения файла: {e}]")
                    dump.write("\n")

if __name__ == "__main__":
    # Укажите путь к вашей папке ('.' — текущая папка)
    project_dir = '.' 
    output_name = 'code_dump.txt'
    
    create_dump(project_dir, output_name)
    print(f"Готово! Весь код собран в {output_name}")
