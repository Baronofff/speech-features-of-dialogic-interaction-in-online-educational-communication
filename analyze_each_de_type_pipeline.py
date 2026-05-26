import pandas as pd
import re
from collections import Counter
from typing import List


class DialogicAnalyzer:
    """
    Класс для анализа диалогических единиц (ДЕ) из CSV-файла.
    Выполняет:
    - загрузку и предобработку данных,
    - удаление ненужных столбцов,
    - группировку по типу ДЕ,
    - для обычных столбцов - вывод распределения значений,
    - для текстовых столбцов - вывод топ-5 частотных слов (без однобуквенных).
    """

    def __init__(self, file_path: str, encoding: str = 'utf-8-sig'):
        """
        Инициализация анализатора.

        Parameters
        ----------
        file_path : str
            Путь к CSV-файлу с данными.
        encoding : str, optional
            Кодировка файла (по умолчанию 'utf-8-sig').
        """
        self.file_path = file_path
        self.encoding = encoding
        self.df = None

    def load_data(self) -> None:
        """
        Загружает данные из CSV-файла в атрибут self.df.
        """
        self.df = pd.read_csv(self.file_path, encoding=self.encoding)

    def drop_columns(self, columns_to_drop: List[str]) -> None:
        """
        Удаляет указанные столбцы из DataFrame (если они существуют).

        Parameters
        ----------
        columns_to_drop : List[str]
            Список названий столбцов для удаления.
        """
        existing = [c for c in columns_to_drop if c in self.df.columns]
        if existing:
            self.df = self.df.drop(columns=existing)

    def preprocess(self) -> None:
        """
        Выполняет всю предобработку: удаляет ненужные столбцы согласно заданию.
        """
        cols_to_drop = [
            'Наличие синтаксического параллелизма',
            'Тип связи между стимулом и реакцией',
            'Способ связи реплик',
            'Наличие вербальной компенсации невербального контакта',
            'Наличие наложений или разрывов коммуникации'
        ]
        self.drop_columns(cols_to_drop)

    @staticmethod
    def tokenize(text) -> List[str]:
        """
        Разбивает текст на слова, оставляя только слова из двух и более букв
        (русские или английские). Однобуквенные слова игнорируются.

        Parameters
        ----------
        text : любой тип, обычно str или NaN
            Входной текст.

        Returns
        -------
        List[str]
            Список слов длиной >= 2.
        """
        if pd.isna(text):
            return []
        text = str(text).lower()
        words = re.findall(r"[а-яёa-z]{2,}", text)
        return words

    def analyze(self) -> None:
        """
        Основной метод анализа: группирует данные по типу ДЕ,
        выводит для каждого типа:
          - для обычных столбцов: распределение значений (value_counts) с процентами,
          - для текстовых столбцов: топ-5 слов (по частоте) с исключением однобуквенных.
        """
        exclude_from_output = ['Идентификатор ДЕ', 'Тип ДЕ', 'ДЕ']

        text_cols = [
            'Ключевые слова-маркеры',
            'Микротема',
            'Конструктивные средства связи',
            'Характеристика реплики-реакции'
        ]

        other_cols = [c for c in self.df.columns if c not in exclude_from_output + text_cols]

        for tipo, group in self.df.groupby('Тип ДЕ'):
            print(f'\n=== Тип ДЕ: {tipo} (всего строк: {len(group)}) ===')
            total = len(group)

            for col in other_cols:
                if col not in self.df.columns:
                    continue
                print(f'\nСтолбец: {col}')
                counts = group[col].value_counts(dropna=False)
                for val, cnt in counts.items():
                    pct = cnt / total * 100
                    print(f'  {val}: {cnt} ({pct:.1f}%)')

            for col in text_cols:
                if col not in self.df.columns:
                    continue
                all_words = []
                for val in group[col]:
                    all_words.extend(self.tokenize(val))
                if not all_words:
                    print(f'\n{col}: нет данных')
                    continue
                counter = Counter(all_words)
                top5 = counter.most_common(5)
                print(f'\n{col} — топ-5 слов:')
                for word, freq in top5:
                    print(f'  {word}: {freq}')

    def run(self) -> None:
        """
        Запускает полный пайплайн: загрузка -> предобработка -> анализ.
        """
        self.load_data()
        self.preprocess()
        self.analyze()


if __name__ == '__main__':
    analyzer = DialogicAnalyzer('materials/dialogic_units.csv')
    analyzer.run()