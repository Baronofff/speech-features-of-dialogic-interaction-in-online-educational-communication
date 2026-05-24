import pandas as pd
import xml.etree.ElementTree as ET
import re
from collections import Counter

def count_words(text):
    if not text:
        return 0
    return len(re.findall(r'\b\w+\b', text, flags=re.UNICODE))

def split_sentences(text):
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]

def my_mean(lst):
    if not lst:
        return 0
    return sum(lst) / len(lst)

def my_median(lst):
    if not lst:
        return 0
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid-1] + s[mid]) / 2
    else:
        return s[mid]


class XMLAnalyzer:
    """Анализ XML-файла с семинарами."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.total_seminars = 0
        self.total_dus = 0
        self.total_turns = 0
        self.total_words = 0
        self.total_sentences = 0
        self.sentence_lengths = []
        self.turn_lengths = []
        self.de_word_counts = []
        self.de_turn_counts = []

    def analyze(self):
        try:
            tree = ET.parse(self.filepath)
            root = tree.getroot()
            files = root.findall('file')
            self.total_seminars = len(files)

            for f in files:
                dus = f.findall('.//dialogic-unit')
                self.total_dus += len(dus)

                for du in dus:
                    turns = du.findall('turn')
                    de_turn_cnt = len(turns)
                    de_word_cnt = 0

                    for turn in turns:
                        text = turn.text if turn.text else ''
                        w = count_words(text)
                        de_word_cnt += w
                        self.total_words += w
                        self.total_turns += 1
                        self.turn_lengths.append(w)

                        sents = split_sentences(text)
                        self.total_sentences += len(sents)
                        for sent in sents:
                            self.sentence_lengths.append(count_words(sent))

                    if de_word_cnt > 0:
                        self.de_word_counts.append(de_word_cnt)
                        self.de_turn_counts.append(de_turn_cnt)

            return True
        except Exception as e:
            print(f"Ошибка при чтении XML: {e}")
            return False

    def print_stats(self):
        print("=" * 60)
        print("СТАТИСТИКА ПО СЕМИНАРАМ (XML)")
        print("=" * 60)
        print(f"Количество семинаров (файлов): {self.total_seminars}")
        print(f"Размеченных диалогических единиц (ДЕ): {self.total_dus}")
        print(f"Всего реплик (turn): {self.total_turns}")
        print(f"Всего слов: {self.total_words}")
        print(f"Всего предложений: {self.total_sentences}")

        if self.total_turns:
            print(f"Среднее слов на реплику: {my_mean(self.turn_lengths):.2f}")
        if self.total_sentences:
            print(f"Средняя длина предложения (слов): {self.total_words / self.total_sentences:.2f}")
            print(f"Медианная длина предложения (слов): {my_median(self.sentence_lengths):.2f}")

        if self.de_word_counts:
            print(f"Среднее количество слов в ДЕ: {my_mean(self.de_word_counts):.2f}")
            print(f"Медианное количество слов в ДЕ: {my_median(self.de_word_counts):.2f}")
            print(f"Среднее количество реплик в ДЕ: {my_mean(self.de_turn_counts):.2f}")
            print(f"Медианное количество реплик в ДЕ: {my_median(self.de_turn_counts):.2f}")

        if self.turn_lengths:
            print(f"Средняя длина реплики (слов): {my_mean(self.turn_lengths):.2f}")
            print(f"Медианная длина реплики (слов): {my_median(self.turn_lengths):.2f}")


class CSVAnalyzer:
    """Анализ CSV-файла с диалогическими единицами."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.total_rows = 0

    def analyze(self):
        try:
            self.df = pd.read_csv(self.filepath, encoding='utf-8')
            self.total_rows = len(self.df)
            return True
        except Exception as e:
            print(f"Ошибка при чтении CSV: {e}")
            return False

    def print_stats(self):
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ПО ДИАЛОГИЧЕСКИМ ЕДИНИЦАМ (CSV)")
        print("=" * 60)
        print(f"Общее количество ДЕ в CSV: {self.total_rows}")

        for col in self.df.columns:
            if col in ['Идентификатор ДЕ', 'ДЕ', 'Микротема', 'Ключевые слова-маркеры']:
                uniq = self.df[col].nunique()
                if col == 'ДЕ':
                    avg_len = self.df[col].astype(str).str.len().mean()
                    print(f"\n{col}: уникальных {uniq}, средняя длина строки {avg_len:.1f} симв.")
                else:
                    print(f"\n{col}: уникальных {uniq}")
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"\n{col}:")
                print(f"  Среднее: {self.df[col].mean():.2f}")
                print(f"  Медиана: {self.df[col].median():.2f}")
                print(f"  Минимум: {self.df[col].min()}, максимум: {self.df[col].max()}")
                continue

            if col == 'Конструктивные средства связи':
                all_means = []
                for val in self.df[col].dropna():
                    all_means.extend([x.strip() for x in str(val).split(',')])
                cnt = Counter(all_means)
                print(f"\n{col} (топ-5):")
                for k, v in cnt.most_common(5):
                    print(f"  {k}: {v}")
                continue

            val_counts = self.df[col].value_counts()
            if len(val_counts) > 20:
                print(f"\n{col}: слишком много категорий ({len(val_counts)}), топ-5:")
                for v, c in val_counts.head(5).items():
                    print(f"  {v}: {c} ({c/self.total_rows*100:.1f}%)")
            else:
                if len(val_counts) > 1 or val_counts.sum() > 0:
                    print(f"\n{col}:")
                    for v, c in val_counts.items():
                        print(f"  {v}: {c} ({c/self.total_rows*100:.1f}%)")

    def print_additional_stats(self):
        """Вывод дополнительных интересных показателей (типы, реакции, позиции, функции, маркеры)."""
        print("\n" + "=" * 60)
        print("ДОПОЛНИТЕЛЬНЫЕ ИНТЕРЕСНЫЕ ПОКАЗАТЕЛИ")
        print("=" * 60)

        if 'Тип ДЕ' in self.df.columns:
            print("\nКлючевые типы диалогических единиц:")
            for t, cnt in self.df['Тип ДЕ'].value_counts().items():
                print(f"  {t}: {cnt} ({cnt/self.total_rows*100:.1f}%)")

        if 'Характеристика реплики-стимула' in self.df.columns:
            print("\nТипы реплик-стимулов:")
            for s, cnt in self.df['Характеристика реплики-стимула'].value_counts().items():
                print(f"  {s}: {cnt} ({cnt/self.total_rows*100:.1f}%)")

        if 'Характеристика реплики-реакции' in self.df.columns:
            all_reactions = []
            for val in self.df['Характеристика реплики-реакции'].dropna():
                all_reactions.extend([r.strip() for r in str(val).split(',')])
            react_counts = Counter(all_reactions)
            print("\nТипы реплик-реакций (объединённые):")
            for r, cnt in react_counts.most_common():
                print(f"  {r}: {cnt}")

        if 'Позиция ДЕ в семинаре' in self.df.columns:
            print("\nРаспределение ДЕ по позиции в семинаре:")
            for p, cnt in self.df['Позиция ДЕ в семинаре'].value_counts().items():
                print(f"  {p}: {cnt} ({cnt/self.total_rows*100:.1f}%)")

        if 'Функция в структуре семинара' in self.df.columns:
            print("\nФункции ДЕ в семинаре:")
            for f, cnt in self.df['Функция в структуре семинара'].value_counts().items():
                print(f"  {f}: {cnt} ({cnt/self.total_rows*100:.1f}%)")

        for feature in ['Наличие эллипсиса', 'Наличие синтаксического параллелизма',
                        'Наличие маркеров хезитации', 'Наличие вербальной компенсации невербального контакта',
                        'Наличие наложений или разрывов коммуникации']:
            if feature in self.df.columns:
                cnt = self.df[feature].value_counts()
                yes = cnt.get('да', 0)
                no = cnt.get('нет', 0)
                print(f"\n{feature}: да – {yes} ({yes/self.total_rows*100:.1f}%), нет – {no} ({no/self.total_rows*100:.1f}%)")


class Pipeline:
    """Пайплайн: последовательно запускает анализ XML и CSV, выводит результаты."""
    def __init__(self, xml_path, csv_path):
        self.xml_analyzer = XMLAnalyzer(xml_path)
        self.csv_analyzer = CSVAnalyzer(csv_path)

    def run(self):
        if self.xml_analyzer.analyze():
            self.xml_analyzer.print_stats()
        else:
            print("Анализ XML не выполнен из-за ошибки.")

        if self.csv_analyzer.analyze():
            self.csv_analyzer.print_stats()
            self.csv_analyzer.print_additional_stats()
        else:
            print("Анализ CSV не выполнен из-за ошибки.")


if __name__ == "__main__":
    pipeline = Pipeline('materials/research_corpus_seminars.xml', 'materials/dialogic_units.csv')
    pipeline.run()