import pandas as pd
import xml.etree.ElementTree as ET
import re
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as ss


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

        self.de_positions = {'начало': 0, 'середина': 0, 'конец': 0}

    @staticmethod
    def _classify_position(index, total):
        if total == 0:
            return None
        ratio = index / total
        if ratio < 1/3:
            return 'начало'
        elif ratio < 2/3:
            return 'середина'
        else:
            return 'конец'

    def analyze(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        files = root.findall('file')
        self.total_seminars = len(files)

        for f in files:
            dus = f.findall('.//dialogic-unit')
            total_in_file = len(dus)
            self.total_dus += total_in_file

            for idx, du in enumerate(dus):
                zone = self._classify_position(idx, total_in_file)
                if zone:
                    self.de_positions[zone] += 1

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

        print("\n" + "=" * 60)
        print("РАСПРЕДЕЛЕНИЕ ДИАЛОГИЧЕСКИХ ЕДИНИЦ ПО ПОЗИЦИИ В СЕМИНАРЕ")
        print("=" * 60)
        for zone in ['начало', 'середина', 'конец']:
            cnt = self.de_positions.get(zone, 0)
            print(f"ДЕ в {zone} семинаров: {cnt}")



class CSVAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.total_rows = 0

    def analyze(self):
        self.df = pd.read_csv(self.filepath, encoding='utf-8')
        self.total_rows = len(self.df)

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



class CorrelationAnalyzer:
    def __init__(self, df, ignore_cols=None):
        self.df = df
        self.ignore_cols = ignore_cols or ['Идентификатор ДЕ', 'ДЕ', 'Ключевые слова-маркеры',
                                            'Микротема', 'Конструктивные средства связи',
                                            'Характеристика реплики-реакции']
        self.df_clean = None
        self.cramers_matrix = None

    @staticmethod
    def _cramers_v(confusion_matrix):
        if confusion_matrix.shape[0] <= 1 or confusion_matrix.shape[1] <= 1:
            return 0.0
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        if n == 0:
            return 0.0
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        denom = min((kcorr-1), (rcorr-1))
        if denom == 0:
            return 0.0
        return np.sqrt(phi2corr / denom)

    def _prepare_data(self):
        features = [c for c in self.df.columns if c not in self.ignore_cols]
        self.df_clean = self.df[features].copy()
        for col in self.df_clean.columns:
            if self.df_clean[col].dtype == 'object':
                le = LabelEncoder()
                self.df_clean[col] = le.fit_transform(self.df_clean[col].astype(str))

    def compute_correlations(self):
        self._prepare_data()
        cols = self.df_clean.columns
        n = len(cols)
        mat = np.zeros((n, n))
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i == j:
                    mat[i, j] = 1.0
                else:
                    crosstab = pd.crosstab(self.df_clean[col1], self.df_clean[col2])
                    mat[i, j] = self._cramers_v(crosstab.values)
        self.cramers_matrix = mat
        self._feature_names = list(cols)

    def plot_heatmap(self, save_path='cramers_v_heatmap.png'):
        if self.cramers_matrix is None:
            self.compute_correlations()
        plt.figure(figsize=(14, 12))
        sns.heatmap(self.cramers_matrix, xticklabels=self._feature_names, yticklabels=self._feature_names,
                    annot=False, cmap='RdBu_r', center=0.5, vmin=0, vmax=1)
        plt.title("Корреляции между признаками (Cramér's V)", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Тепловая карта корреляций сохранена в {save_path}")

    def print_top_correlations(self, threshold=0.5, top_n=5):
        if self.cramers_matrix is None:
            self.compute_correlations()
        n = len(self._feature_names)
        corr_list = []
        for i in range(n):
            for j in range(i+1, n):
                v = self.cramers_matrix[i, j]
                if v > threshold:
                    corr_list.append((self._feature_names[i], self._feature_names[j], v))
        corr_list.sort(key=lambda x: -x[2])
        print("\n=== Топ-5 сильных корреляций (Cramér's V > 0.5) ===")
        for a, b, v in corr_list[:top_n]:
            print(f"{a} <-> {b}: {v:.3f}")

    def run(self):
        self.compute_correlations()
        self.plot_heatmap()
        self.print_top_correlations()



class ClusteringAnalyzer:
    def __init__(self, df, ignore_cols=None):
        self.df = df
        self.ignore_cols = ignore_cols or ['Идентификатор ДЕ', 'ДЕ', 'Ключевые слова-маркеры',
                                            'Микротема', 'Конструктивные средства связи',
                                            'Характеристика реплики-реакции']
        self.best_k = None
        self.clusters = None
        self.X_scaled = None

    def _prepare_features(self):
        features = [c for c in self.df.columns if c not in self.ignore_cols]
        df_features = self.df[features].copy()
        df_encoded = pd.get_dummies(df_features, drop_first=False)
        print(f"Размерность one-hot матрицы: {df_encoded.shape}")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(df_encoded)
        return self.X_scaled

    def find_optimal_k(self, k_range=range(2, 10)):
        X = self._prepare_features()
        sil_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            sil_scores.append(sil)
            print(f"k={k}: silhouette = {sil:.4f}")
        self.best_k = k_range[np.argmax(sil_scores)]
        best_sil = max(sil_scores)
        print(f"\nВыбрано k = {self.best_k} (силуэт = {best_sil:.4f})")

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, sil_scores, 'o-', color='b')
        plt.axvline(x=self.best_k, linestyle='--', color='r', label=f'Лучшее k={self.best_k}')
        plt.xlabel('Число кластеров')
        plt.ylabel('Средний силуэтный коэффициент')
        plt.title('Выбор числа кластеров по силуэту')
        plt.legend()
        plt.grid(True)
        plt.savefig('silhouette_score_plot.png', dpi=150)
        plt.close()
        print("График силуэта сохранён в silhouette_score_plot.png")

    def cluster(self):
        if self.X_scaled is None:
            self._prepare_features()
        if self.best_k is None:
            self.find_optimal_k()
        kmeans = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.X_scaled)
        df_result = self.df.copy()
        df_result['Cluster'] = self.clusters
        df_result.to_csv('dialogic_units_clustered.csv', index=False, encoding='utf-8-sig')
        print("Результат кластеризации сохранён в 'dialogic_units_clustered.csv'")
        return self.clusters

    def pca_visualize(self):
        if self.X_scaled is None or self.clusters is None:
            self.cluster()
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        expl_var = pca.explained_variance_ratio_
        print(f"Доля объяснённой дисперсии первыми двумя компонентами PCA: {expl_var[0]:.3f} + {expl_var[1]:.3f} = {sum(expl_var):.3f}")

        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = self.clusters
        df_pca['Type'] = self.df['Тип ДЕ']

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1', style='Type', s=60)
        plt.title(f'Кластеризация ДЕ (PCA, k={self.best_k})')
        plt.xlabel(f'PC1 ({expl_var[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({expl_var[1]*100:.1f}%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('clusters_pca.png', dpi=150)
        plt.close()
        print("PCA-визуализация кластеров сохранена в clusters_pca.png")

    def print_cluster_composition(self):
        if self.clusters is None:
            self.cluster()
        df_temp = self.df.copy()
        df_temp['Cluster'] = self.clusters
        cross = pd.crosstab(df_temp['Cluster'], df_temp['Тип ДЕ'], normalize='index')
        print("\n=== Распределение типов ДЕ по кластерам (по строкам) ===")
        print(cross.round(3))

        cross.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title('Состав кластеров по прагматическим типам ДЕ')
        plt.xlabel('Кластер')
        plt.ylabel('Доля')
        plt.legend(title='Тип ДЕ', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('cluster_composition.png', dpi=150)
        plt.close()
        print("Диаграмма состава кластеров сохранена в cluster_composition.png")

    def run(self):
        self.find_optimal_k()
        self.cluster()
        self.pca_visualize()
        self.print_cluster_composition()


class BasicVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_distributions(self, save_path='basic_distributions.png'):
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        self.df['Тип ДЕ'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Распределение ДЕ по прагматическим типам')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 2)
        self.df['Позиция ДЕ в семинаре'].value_counts().plot(kind='bar', color='salmon')
        plt.title('Позиция ДЕ в семинаре')

        plt.subplot(2, 2, 3)
        self.df['Функция в структуре семинара'].value_counts().head(8).plot(kind='bar', color='lightgreen')
        plt.title('Функция ДЕ в семинаре')
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 4)
        self.df['Наличие эллипсиса'].value_counts().plot(kind='bar', color='gold')
        plt.title('Наличие эллипсиса')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Базовые распределения сохранены в {save_path}")

    def run(self):
        self.plot_distributions()



class Pipeline:
    def __init__(self, xml_path, csv_path):
        self.xml_path = xml_path
        self.csv_path = csv_path

    def run(self):
        xml_analyzer = XMLAnalyzer(self.xml_path)
        xml_analyzer.analyze()
        xml_analyzer.print_stats()

        csv_analyzer = CSVAnalyzer(self.csv_path)
        csv_analyzer.analyze()
        csv_analyzer.print_stats()
        csv_analyzer.print_additional_stats()

        corr_analyzer = CorrelationAnalyzer(csv_analyzer.df)
        corr_analyzer.run()

        clust_analyzer = ClusteringAnalyzer(csv_analyzer.df)
        clust_analyzer.run()

        viz = BasicVisualizer(csv_analyzer.df)
        viz.run()

        print("\n" + "=" * 60)
        print("ВСЕ АНАЛИЗЫ УСПЕШНО ЗАВЕРШЕНЫ")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = Pipeline('materials/research_corpus_seminars.xml', 'materials/dialogic_units.csv')
    pipeline.run()