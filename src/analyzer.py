"""
Модуль для анализа отзывов.
Содержит основной класс DiplomReviewAnalyzer.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
import pymorphy2
from tqdm import tqdm
import json
import os

# Загрузка стоп-слов при импорте
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DiplomReviewAnalyzer:
    """
    Анализатор отзывов для дипломного проекта.
    Позволяет загружать данные из CSV, анализировать тексты отзывов,
    распределять слова по тональности и категориям.
    """
    
    def __init__(self, use_tqdm=True, custom_stopwords_file=None):
        """
        Инициализация анализатора.
        
        Parameters:
        use_tqdm (bool): Использовать прогресс-бар при обработке
        custom_stopwords_file (str): Путь к файлу с кастомными стоп-словами
        """
        self.morph = pymorphy2.MorphAnalyzer()
        self.use_tqdm = use_tqdm
        
        # Стандартные стоп-слова из NLTK
        self.stopwords = set(stopwords.words('russian'))
        
        # Загружаем кастомные стоп-слова, если файл существует
        self.custom_stopwords = self._load_custom_stopwords(custom_stopwords_file)
        self.stopwords.update(self.custom_stopwords)
        
        # Для хранения результатов
        self.reset_results()

    def translate_category(self, category_name):
        """
        Перевод названия категории с английского на русский.
        Использует библиотеку googletrans.
        
        Parameters:
        category_name (str): Название категории на английском
        
        Returns:
        str: Переведенное название или оригинал, если перевод не удался
        """
        # Словарь для быстрого перевода без API
        translation_dict = {
            'smartphones': 'смартфоны',
            'big-home-appl': 'крупная бытовая техника',
            'small-home-appl': 'мелкая бытовая техника',
            'kitchen-home-appl': 'кухонная техника',
            'headphones': 'наушники',
            'beauty': 'красота и уход',
            'tires': 'шины',
            'climate-equipment': 'климатическое оборудование',
            'car-electronics': 'автоэлектроника',
            'perfumes': 'парфюмерия',
            'watches': 'часы',
            'car-audio': 'автозвук',
            'wearables': 'носимая электроника',
            'power-banks': 'внешние аккумуляторы',
            'portable-speakers': 'портативные колонки',
            'books': 'книги',
            'memory-cards': 'карты памяти',
            'unknown': 'без категории'
        }
        
        # Сначала проверяем в словаре
        if category_name.lower() in translation_dict:
            return translation_dict[category_name.lower()]
        
        # Если нет в словаре, пробуем через googletrans
        try:
            from googletrans import Translator
            translator = Translator()
            translated = translator.translate(category_name, src='en', dest='ru')
            return translated.text
        except Exception as e:
            # Если перевод не удался, возвращаем оригинальное название
            return category_name
    
    def _load_custom_stopwords(self, custom_stopwords_file):
        """
        Загрузка кастомных стоп-слов из файла.
        
        Поддерживаемые форматы:
        - .txt (по одному слову на строку)
        - .json (список слов)
        """
        custom_stopwords = set()
        
        if custom_stopwords_file and os.path.exists(custom_stopwords_file):
            try:
                if custom_stopwords_file.endswith('.txt'):
                    with open(custom_stopwords_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip().lower()
                            if word and not word.startswith('#'):
                                custom_stopwords.add(word)
                    print(f"✅ Загружено {len(custom_stopwords)} кастомных стоп-слов из {custom_stopwords_file}")
                
                elif custom_stopwords_file.endswith('.json'):
                    with open(custom_stopwords_file, 'r', encoding='utf-8') as f:
                        words = json.load(f)
                        if isinstance(words, list):
                            custom_stopwords = set([w.lower() for w in words if w])
                            print(f"✅ Загружено {len(custom_stopwords)} кастомных стоп-слов из {custom_stopwords_file}")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке стоп-слов: {e}")
        
        return custom_stopwords
    
    def save_stopwords_to_file(self, filename='custom_stopwords.txt', words=None):
        """
        Сохранение стоп-слов в файл.
        
        Parameters:
        filename (str): Имя файла для сохранения
        words (list): Список слов для сохранения (если None, сохраняются текущие кастомные)
        """
        if words is None:
            words = sorted(self.custom_stopwords)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Кастомные стоп-слова для анализа отзывов\n")
            f.write("# Добавляйте по одному слову на строку\n\n")
            for word in words:
                f.write(f"{word}\n")
        
        print(f"✅ Сохранено {len(words)} стоп-слов в {filename}")
        return filename
    
    def suggest_stopwords_from_results(self, results, top_n=50, min_freq=1000):
        """
        Предложить стоп-слова на основе результатов анализа.
        
        Parameters:
        results (dict): Результаты анализа
        top_n (int): Сколько топ-слов рассмотреть
        min_freq (int): Минимальная частота для предложения
        
        Returns:
        list: Список предложенных стоп-слов
        """
        suggestions = set()
        
        # Слова, которые часто встречаются во всех категориях
        common_words = set()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            freq = results['frequencies'][sentiment]
            for word, count in freq.most_common(top_n):
                if count >= min_freq:
                    common_words.add(word)
        
        # Стоп-слова из NLTK для справки
        nltk_stopwords = set(stopwords.words('russian'))
        
        # Кандидаты: слова, похожие на стоп-слова
        candidates = []
        for word in common_words:
            # Проверяем, похоже ли на стоп-слово
            if word in nltk_stopwords:
                candidates.append((word, "уже в NLTK"))
            elif word in ['очень', 'всё', 'это', 'весь', 'такой', 'который', 'можно', 'мочь']:
                candidates.append((word, "частотное служебное"))
            elif len(word) <= 3:  # короткие слова
                candidates.append((word, "короткое"))
        
        print("\n📋 ПРЕДЛОЖЕНИЯ ПО СТОП-СЛОВАМ:")
        print("-" * 50)
        for word, reason in candidates:
            print(f"  • {word} - {reason}")
        
        return [word for word, _ in candidates]
    
    def reset_results(self):
        """Сброс результатов для нового анализа"""
        # Для хранения результатов по тональности (общие)
        self.words_by_sentiment = {
            'positive': [],
            'negative': [],
            'neutral': []
        }
        
        # Для хранения результатов по категориям
        self.category_words = defaultdict(lambda: {
            'positive': [],
            'negative': [],
            'neutral': []
        })
        
        # Для хранения статистики по полям
        self.field_stats = {
            'text': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
            'plus': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0},
            'minus': {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        }
        
        # Дополнительная статистика
        self.all_words_freq = Counter()
        self.category_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
    
    def load_from_csv(self, filename, encoding=None):
        """Загрузка данных из CSV файла"""
        encodings_to_try = [encoding] if encoding else ['utf-8', 'cp1251', 'latin1']
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(filename, encoding=enc)
                print(f"✅ Загружено {len(df)} отзывов из файла {filename} (кодировка {enc})")
                return df
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"❌ Файл {filename} не найден!")
                return None
            except Exception as e:
                print(f"❌ Ошибка при загрузке файла: {e}")
                return None
        
        print(f"❌ Не удалось прочитать файл {filename} ни в одной из кодировок")
        return None
    
    def clean_text(self, text):
        """Очистка текста от лишних символов"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text).lower()
        # Оставляем русские буквы и пробелы
        text = re.sub(r'[^а-яё\s]', ' ', text)
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def lemmatize_text(self, text):
        """Лемматизация текста"""
        if not text:
            return []
        
        words = text.split()
        result = []
        
        for word in words:
            if len(word) <= 2:
                continue
                
            try:
                parsed = self.morph.parse(word)[0]
                lemma = parsed.normal_form
                
                if len(lemma) > 2 and lemma not in self.stopwords:
                    result.append(lemma)
                    self.all_words_freq[lemma] += 1
            except Exception:
                continue
        
        return result
    
    def add_custom_stopwords(self, words):
        """
        Добавить кастомные стоп-слова.
        
        Parameters:
        words (list): Список слов для добавления
        """
        if isinstance(words, str):
            words = [words]
        
        for word in words:
            self.custom_stopwords.add(word.lower())
            self.stopwords.add(word.lower())
        
        print(f"✅ Добавлено {len(words)} кастомных стоп-слов")
    
    def remove_custom_stopwords(self, words):
        """
        Удалить кастомные стоп-слова.
        
        Parameters:
        words (list): Список слов для удаления
        """
        if isinstance(words, str):
            words = [words]
        
        for word in words:
            word = word.lower()
            if word in self.custom_stopwords:
                self.custom_stopwords.remove(word)
                self.stopwords.remove(word)
        
        print(f"✅ Удалено {len(words)} кастомных стоп-слов")
    
    def analyze_reviews(self, df):
        """Анализ отзывов с учетом всех трех колонок и категорий"""
        print("\n" + "=" * 60)
        print("НАЧАЛО АНАЛИЗА ОТЗЫВОВ")
        print("=" * 60)

        self.reset_results()

        # Проверяем наличие необходимых колонок
        required_columns = ['text', 'plus', 'minus', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ В файле отсутствуют колонки: {missing_columns}")
            return None

        # Проверяем наличие колонки category
        has_category = 'category' in df.columns
        if has_category:
            df['category'] = df['category'].fillna('unknown')

        # ШАГ 1: ФИЛЬТРАЦИЯ ПО ЯЗЫКУ (только russian)
        print("\n🔍 ШАГ 1: Фильтрация по языку...")

        if 'language' in df.columns:
            # Приводим к нижнему регистру для сравнения
            df['language_lower'] = df['language'].astype(str).str.lower().str.strip()

            # Статистика по языкам
            lang_counts = df['language_lower'].value_counts()
            print(f"📊 Распределение языков в данных:")
            for lang, count in lang_counts.head(10).items():
                print(f"   {lang}: {count} ({count/len(df)*100:.1f}%)")

            # Фильтруем ТОЛЬКО russian (без проверок на казахские буквы)
            russian_keywords = ['ru', 'рус', 'russian', 'россия', 'russia']
            russian_mask = df['language_lower'].apply(
                lambda x: any(keyword in x for keyword in russian_keywords)
            )

            df_russian = df[russian_mask].copy()

            print(f"\n✅ После фильтрации по языку:")
            print(f"   Всего отзывов: {len(df_russian)} из {len(df)}")
            print(f"   Исключено нерусских: {len(df) - len(df_russian)}")

            # Просто информируем о наличии других языков в русской выборке, но не фильтруем
            other_langs = df_russian[~df_russian['language_lower'].isin(['ru', 'russian', 'русский'])]['language_lower'].unique()
            if len(other_langs) > 0:
                print(f"   ℹ️  В русской выборке также присутствуют: {', '.join(other_langs)}")
        else:
            print("⚠️ Колонка 'language' не найдена, работаем со всеми данными")
            df_russian = df.copy()

        if len(df_russian) == 0:
            print("❌ Нет русскоязычных отзывов для анализа!")
            return None

        # ШАГ 2: ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ
        print("\n🎯 ШАГ 2: Определение тональности...")
        df_russian['sentiment'] = 'neutral'
        df_russian.loc[df_russian['rating'] >= 4, 'sentiment'] = 'positive'
        df_russian.loc[df_russian['rating'] <= 2, 'sentiment'] = 'negative'

        sentiment_counts = df_russian['sentiment'].value_counts()
        print(f"\nРаспределение по тональности:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} ({count/len(df_russian)*100:.1f}%)")

        # ШАГ 3: ОБРАБОТКА ОТЗЫВОВ
        print(f"\n🔧 ШАГ 3: Обработка текстов (очистка и лемматизация)...")

        iterator = tqdm(df_russian.iterrows(), total=len(df_russian), desc="Прогресс") if self.use_tqdm else df_russian.iterrows()

        for idx, row in iterator:
            self._process_single_review(row, has_category)

        print(f"  ✅ Обработано {len(df_russian)} отзывов. Готово!")

        # ШАГ 4: ПОДГОТОВКА РЕЗУЛЬТАТОВ
        results = self._prepare_results(df_russian, has_category, sentiment_counts)

        # ШАГ 5: ОЧИСТКА ВРЕМЕННЫХ КОЛОНОК
        if 'language_lower' in df_russian.columns:
            df_russian.drop(columns=['language_lower'], inplace=True)
        if 'sentiment' in df_russian.columns:
            df_russian.drop(columns=['sentiment'], inplace=True)

        return results
    
    def _process_single_review(self, row, has_category):
        """Обработка одного отзыва"""
        sentiment = row['sentiment']
        category = row.get('category', 'unknown') if has_category else 'unknown'
        
        # Обновляем статистику по категориям
        self.category_stats[category]['total'] += 1
        self.category_stats[category][sentiment] += 1
        
        # Проверяем наличие текста
        if pd.isna(row['text']) and pd.isna(row['plus']) and pd.isna(row['minus']):
            return
        
        # Обработка колонки text
        self._process_field(row, 'text', sentiment, category)
        
        # Обработка колонки plus
        self._process_plus_field(row, sentiment, category)
        
        # Обработка колонки minus
        self._process_minus_field(row, category)
    
    def _process_field(self, row, field, sentiment, category):
        """Обработка обычного поля"""
        if pd.isna(row[field]) or str(row[field]).strip() == '':
            return
        
        cleaned = self.clean_text(row[field])
        words = self.lemmatize_text(cleaned)
        
        if words:
            self.words_by_sentiment[sentiment].extend(words)
            self.category_words[category][sentiment].extend(words)
            self.field_stats[field][sentiment] += len(words)
            self.field_stats[field]['total'] += 1
    
    def _process_plus_field(self, row, sentiment, category):
        """Специальная обработка поля plus"""
        if pd.isna(row['plus']) or str(row['plus']).strip() == '':
            return
        
        cleaned = self.clean_text(row['plus'])
        words = self.lemmatize_text(cleaned)
        
        if words:
            target_sentiment = 'positive' if sentiment in ['positive', 'neutral'] else 'negative'
            self.words_by_sentiment[target_sentiment].extend(words)
            self.category_words[category][target_sentiment].extend(words)
            self.field_stats['plus'][target_sentiment] += len(words)
            self.field_stats['plus']['total'] += 1
    
    def _process_minus_field(self, row, category):
        """Специальная обработка поля minus"""
        if pd.isna(row['minus']) or str(row['minus']).strip() == '':
            return
        
        cleaned = self.clean_text(row['minus'])
        words = self.lemmatize_text(cleaned)
        
        if words:
            self.words_by_sentiment['negative'].extend(words)
            self.category_words[category]['negative'].extend(words)
            self.field_stats['minus']['negative'] += len(words)
            self.field_stats['minus']['total'] += 1
    
    def _prepare_results(self, df_russian, has_category, sentiment_counts):
        """Подготовка результатов анализа"""
        # Общие частотные словари
        freq_dicts = {
            'positive': Counter(self.words_by_sentiment['positive']),
            'negative': Counter(self.words_by_sentiment['negative']),
            'neutral': Counter(self.words_by_sentiment['neutral'])
        }
        
        # Частотные словари по категориям
        category_freqs = {}
        for category, sentiments in self.category_words.items():
            category_freqs[category] = {
                'positive': Counter(sentiments['positive']),
                'negative': Counter(sentiments['negative']),
                'neutral': Counter(sentiments['neutral'])
            }
        
        return {
            'frequencies': freq_dicts,
            'category_frequencies': category_freqs,
            'df': df_russian,
            'has_category': has_category,
            'stats': {
                'words_count': {
                    'positive': len(self.words_by_sentiment['positive']),
                    'negative': len(self.words_by_sentiment['negative']),
                    'neutral': len(self.words_by_sentiment['neutral'])
                },
                'field_stats': self.field_stats,
                'sentiment_counts': sentiment_counts.to_dict(),
                'category_stats': dict(self.category_stats)
            },
            'all_words': self.all_words_freq
        }
    
    def analyze_categories(self, df, min_reviews=5):
        """
        Анализ категорий товаров с учетом стоп-слов, лемматизацией и прогресс-баром.
        """
        print("\n" + "=" * 60)
        print("📊 АНАЛИЗ КАТЕГОРИЙ ТОВАРОВ")
        print("=" * 60)

        # Проверяем наличие колонки category
        if 'category' not in df.columns:
            print("❌ Колонка 'category' не найдена в данных")
            return None

        # Заполняем пропущенные категории
        df['category'] = df['category'].fillna('unknown')

        # Базовая статистика по категориям
        category_counts = df['category'].value_counts()

        # Выводим статистику по unknown отдельно
        unknown_count = category_counts.get('unknown', 0)
        print(f"\n📌 Всего уникальных категорий: {len(category_counts)}")
        print(f"📌 Отзывов без категории (unknown): {unknown_count}")
        print(f"📌 Загружено стоп-слов: {len(self.stopwords)}")

        # Фильтруем категории с достаточным количеством отзывов
        # ИСКЛЮЧАЕМ 'unknown' из списка для анализа
        valid_categories = [cat for cat in category_counts[category_counts >= min_reviews].index.tolist() 
                            if cat != 'unknown']

        print(f"\n📌 Категорий с >= {min_reviews} отзывами (без unknown): {len(valid_categories)}")
        if unknown_count > 0:
            print(f"   ℹ️  Категория 'unknown' исключена из анализа (неинформативна)")
        print(f"\n⏳ Начинаю обработку категорий...\n")

        # Собираем детальную статистику по каждой категории (только валидные)
        categories_stats = {}

        # Используем tqdm для прогресс-бара
        for category in tqdm(valid_categories, desc="📊 Обработка категорий", unit="кат"):
            category_df = df[df['category'] == category]

            # Статистика по рейтингам
            rating_counts = category_df['rating'].value_counts().sort_index()

            # Тональность
            sentiment_counts = {
                'positive': len(category_df[category_df['rating'] >= 4]),
                'neutral': len(category_df[category_df['rating'] == 3]),
                'negative': len(category_df[category_df['rating'] <= 2])
            }

            # Средний рейтинг
            avg_rating = category_df['rating'].mean()

            # Заполненность полей
            fields_filled = {
                'text': category_df['text'].notna().sum(),
                'plus': category_df['plus'].notna().sum(),
                'minus': category_df['minus'].notna().sum()
            }

            # Процент заполнения
            total = len(category_df)
            fields_percent = {
                field: (count / total * 100) for field, count in fields_filled.items()
            }

            # АНАЛИЗ ДЛИНЫ ОТЗЫВОВ (общий)
            texts = category_df['text'].dropna().astype(str)
            if len(texts) > 0:
                text_lengths = texts.str.len()
                min_length = text_lengths.min()
                max_length = text_lengths.max()
                avg_length = text_lengths.mean()
            else:
                min_length = max_length = avg_length = 0

            # АНАЛИЗ ДЛИНЫ ПОЛЯ PLUS
            plus_texts = category_df['plus'].dropna().astype(str)
            if len(plus_texts) > 0:
                plus_lengths = plus_texts.str.len()
                min_plus = plus_lengths.min()
                max_plus = plus_lengths.max()
                avg_plus = plus_lengths.mean()
            else:
                min_plus = max_plus = avg_plus = 0

            # АНАЛИЗ ДЛИНЫ ПОЛЯ MINUS
            minus_texts = category_df['minus'].dropna().astype(str)
            if len(minus_texts) > 0:
                minus_lengths = minus_texts.str.len()
                min_minus = minus_lengths.min()
                max_minus = minus_lengths.max()
                avg_minus = minus_lengths.mean()
            else:
                min_minus = max_minus = avg_minus = 0

            # АНАЛИЗ ДЛИНЫ ОТЗЫВОВ ПО ТОНАЛЬНОСТИ
            positive_texts = category_df[category_df['rating'] >= 4]['text'].dropna().astype(str)
            neutral_texts = category_df[category_df['rating'] == 3]['text'].dropna().astype(str)
            negative_texts = category_df[category_df['rating'] <= 2]['text'].dropna().astype(str)

            # Рассчитываем длину для положительных отзывов
            if len(positive_texts) > 0:
                positive_lengths = positive_texts.str.len()
                avg_positive = positive_lengths.mean()
                min_positive = positive_lengths.min()
                max_positive = positive_lengths.max()
            else:
                avg_positive = min_positive = max_positive = 0

            # Рассчитываем длину для нейтральных отзывов
            if len(neutral_texts) > 0:
                neutral_lengths = neutral_texts.str.len()
                avg_neutral = neutral_lengths.mean()
                min_neutral = neutral_lengths.min()
                max_neutral = neutral_lengths.max()
            else:
                avg_neutral = min_neutral = max_neutral = 0

            # Рассчитываем длину для отрицательных отзывов
            if len(negative_texts) > 0:
                negative_lengths = negative_texts.str.len()
                avg_negative = negative_lengths.mean()
                min_negative = negative_lengths.min()
                max_negative = negative_lengths.max()
            else:
                avg_negative = min_negative = max_negative = 0

            # Топ слов из полей plus и minus
            top_plus_words = []
            top_minus_words = []

            if fields_filled['plus'] > 0:
                all_plus = ' '.join(category_df['plus'].dropna().astype(str))
                cleaned = self.clean_text(all_plus)
                words = self.lemmatize_text(cleaned)
                word_counts = Counter(words)
                top_plus_words = word_counts.most_common(10)

            if fields_filled['minus'] > 0:
                all_minus = ' '.join(category_df['minus'].dropna().astype(str))
                cleaned = self.clean_text(all_minus)
                words = self.lemmatize_text(cleaned)
                word_counts = Counter(words)
                top_minus_words = word_counts.most_common(10)

            # Сохраняем все данные в словарь (только для валидных категорий)
            categories_stats[category] = {
                'total_reviews': total,
                'avg_rating': round(avg_rating, 2),
                'rating_distribution': rating_counts.to_dict(),
                'sentiment': sentiment_counts,
                'sentiment_percent': {
                    'positive': round(sentiment_counts['positive'] / total * 100, 1),
                    'neutral': round(sentiment_counts['neutral'] / total * 100, 1),
                    'negative': round(sentiment_counts['negative'] / total * 100, 1)
                },
                'fields_filled': fields_filled,
                'fields_percent': fields_percent,
                'top_plus_words': top_plus_words,
                'top_minus_words': top_minus_words,
                'text_length': {
                    'min': min_length,
                    'max': max_length,
                    'avg': round(avg_length, 0)
                },
                'plus_length': {
                    'min': min_plus,
                    'max': max_plus,
                    'avg': round(avg_plus, 0)
                },
                'minus_length': {
                    'min': min_minus,
                    'max': max_minus,
                    'avg': round(avg_minus, 0)
                },
                'positive_length': {
                    'min': min_positive,
                    'max': max_positive,
                    'avg': round(avg_positive, 0),
                    'count': len(positive_texts)
                },
                'neutral_length': {
                    'min': min_neutral,
                    'max': max_neutral,
                    'avg': round(avg_neutral, 0),
                    'count': len(neutral_texts)
                },
                'negative_length': {
                    'min': min_negative,
                    'max': max_negative,
                    'avg': round(avg_negative, 0),
                    'count': len(negative_texts)
                }
            }

        print()  # Пустая строка после прогресс-бара

        # Проверяем, есть ли валидные категории для вывода
        if not categories_stats:
            print("❌ Нет категорий с достаточным количеством отзывов для анализа!")
            return None

        # Вывод результатов (только валидные категории)
        self._print_categories_stats(categories_stats, valid_categories)

        return categories_stats

    
    
    def _print_categories_stats(self, categories_stats, valid_categories):
        """Вывод статистики по категориям (только валидные категории)"""

        print("\n" + "=" * 60)
        print("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ПО КАТЕГОРИЯМ (С ЛЕММАТИЗАЦИЕЙ)")
        print("=" * 60)

        # Сортируем категории по количеству отзывов
        sorted_categories = sorted(
            categories_stats.items(),
            key=lambda x: x[1]['total_reviews'],
            reverse=True
        )

        for i, (category, stats) in enumerate(sorted_categories, 1):
            # Переводим название категории
            translated_category = self.translate_category(category)
            print(f"\n{i:2d}. 📍 {translated_category.upper()}")
            print(f"   📊 Всего отзывов: {stats['total_reviews']}")
            print(f"   ⭐ Средний рейтинг: {stats['avg_rating']}")

            # Длина отзывов
            print(f"   📝 Длина отзывов:")
            print(f"      text: мин {stats['text_length']['min']:.0f} | ср {stats['text_length']['avg']:.0f} | макс {stats['text_length']['max']:.0f} симв.")
            print(f"      plus: мин {stats['plus_length']['min']:.0f} | ср {stats['plus_length']['avg']:.0f} | макс {stats['plus_length']['max']:.0f} симв.")
            print(f"      minus: мин {stats['minus_length']['min']:.0f} | ср {stats['minus_length']['avg']:.0f} | макс {stats['minus_length']['max']:.0f} симв.")

            # Тональность
            pos = stats['sentiment_percent']['positive']
            neu = stats['sentiment_percent']['neutral']
            neg = stats['sentiment_percent']['negative']
            print(f"   📈 Тональность: 🟢 {pos}% | ⚪ {neu}% | 🔴 {neg}%")

            # Заполненность полей
            text_pct = stats['fields_percent']['text']
            plus_pct = stats['fields_percent']['plus']
            minus_pct = stats['fields_percent']['minus']
            print(f"   📊 Заполненность: text {text_pct:.0f}% | plus {plus_pct:.0f}% | minus {minus_pct:.0f}%")

            # Топ слова из plus
            if stats['top_plus_words']:
                plus_words = ', '.join([f"{w}({c})" for w, c in stats['top_plus_words'][:5]])
                print(f"   👍 Часто в PLUS: {plus_words}")
            else:
                print(f"   👍 Часто в PLUS: нет данных")

            # Топ слова из minus
            if stats['top_minus_words']:
                minus_words = ', '.join([f"{w}({c})" for w, c in stats['top_minus_words'][:5]])
                print(f"   👎 Часто в MINUS: {minus_words}")
            else:
                print(f"   👎 Часто в MINUS: нет данных")

        # Вывод категорий с наибольшим потенциалом для улучшения
        print("\n" + "=" * 60)
        print("🎯 КАТЕГОРИИ ДЛЯ УЛУЧШЕНИЯ (точки роста)")
        print("=" * 60)

        growth_categories = sorted(
            [(cat, stats) for cat, stats in categories_stats.items() 
             if stats['total_reviews'] >= 10],
            key=lambda x: x[1]['sentiment']['negative'] / x[1]['total_reviews'],
            reverse=True
        )

        for i, (category, stats) in enumerate(growth_categories, 1):
            translated_category = self.translate_category(category)
            neg_share = stats['sentiment']['negative'] / stats['total_reviews'] * 100
            print(f"{i:2d}. {translated_category[:30]:30} 🔴 {neg_share:5.1f}% негатива ({stats['sentiment']['negative']}/{stats['total_reviews']})")

            if stats['top_minus_words']:
                minus_words = ', '.join([w for w, _ in stats['top_minus_words'][:3]])
                print(f"     Часто: {minus_words}")

        # Вывод самых популярных категорий
        print("\n" + "=" * 60)
        print("🔥 САМЫЕ ПОПУЛЯРНЫЕ КАТЕГОРИИ")
        print("=" * 60)

        popular = sorted(
            categories_stats.items(),
            key=lambda x: x[1]['total_reviews'],
            reverse=True
        )

        for i, (category, stats) in enumerate(popular, 1):
            translated_category = self.translate_category(category)
            print(f"{i:2d}. {translated_category[:40]:40} 📊 {stats['total_reviews']} отзывов | ⭐ {stats['avg_rating']} | 📝 text: {stats['text_length']['avg']:.0f} сим.")