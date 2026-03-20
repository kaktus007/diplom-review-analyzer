"""
Модуль для визуализации результатов анализа.
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from analyzer import DiplomReviewAnalyzer


class ReviewVisualizer:
    """Класс для визуализации результатов анализа отзывов"""
    
    def __init__(self, output_dir='output/images'):
        """
        Parameters:
        output_dir (str): Папка для сохранения изображений
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        # Создаем экземпляр анализатора для перевода
        self.analyzer = DiplomReviewAnalyzer(use_tqdm=False)
    
    def create_wordcloud(self, freq_dict, title, filename, color='viridis'):
        """
        Создать облако слов.
        
        Parameters:
        freq_dict (Counter): Частотный словарь
        title (str): Заголовок
        filename (str): Имя файла для сохранения
        color (str): Цветовая схема
        """
        if not freq_dict:
            print(f"❌ Нет данных для {title}")
            return
        
        # Берем топ-50 слов
        top_words = dict(freq_dict.most_common(50))
        
        wordcloud = WordCloud(
            width=1000, 
            height=500,
            background_color='white',
            max_words=50,
            colormap=color,
            random_state=42,
            contour_width=1,
            contour_color='steelblue'
        ).generate_from_frequencies(top_words)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        
        full_path = f"{self.output_dir}/{filename}"
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: {full_path}")
    
    def plot_rating_distribution(self, df, filename='rating_distribution.png'):
        """График распределения рейтингов"""
        plt.figure(figsize=(10, 6))
        
        rating_counts = df['rating'].value_counts().sort_index()
        bars = plt.bar(rating_counts.index, rating_counts.values, 
                      color='skyblue', edgecolor='navy')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Распределение оценок в отзывах', fontsize=14)
        plt.xlabel('Рейтинг')
        plt.ylabel('Количество отзывов')
        plt.xticks(range(1, 6))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        full_path = f"{self.output_dir}/{filename}"
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"   ✅ Сохранено: {full_path}")
    
    def plot_field_usage(self, df, filename='field_usage.png'):
        """График использования полей"""
        plt.figure(figsize=(10, 6))
        
        text_filled = df['text'].notna().sum()
        plus_filled = df['plus'].notna().sum()
        minus_filled = df['minus'].notna().sum()
        
        fields = ['text', 'plus', 'minus']
        counts = [text_filled, plus_filled, minus_filled]
        colors = ['lightgreen', 'gold', 'lightcoral']
        
        bars = plt.bar(fields, counts, color=colors, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)} ({height/len(df)*100:.1f}%)', 
                    ha='center', va='bottom')
        
        plt.title('Использование полей в отзывах', fontsize=14)
        plt.xlabel('Поле')
        plt.ylabel('Количество заполненных записей')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        full_path = f"{self.output_dir}/{filename}"
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"   ✅ Сохранено: {full_path}")
    
    def create_category_chart(self, results, filename='category_sentiment.png'):
        """График тональности по категориям"""
        if not results['has_category']:
            return
        
        category_stats = results['stats']['category_stats']
        
        # Получаем все категории
        all_categories = [(cat, stats) for cat, stats in category_stats.items() if cat != 'unknown']
        
        # Сортируем по количеству отзывов и берем топ-10 для читаемости графика
        top_categories = sorted(
            all_categories,
            key=lambda x: x[1]['total'],
            reverse=True
        )[:10]
        
        if not top_categories:
            return
        
        # Переводим названия категорий
        categories = [self.analyzer.translate_category(cat) for cat, _ in top_categories]
        positive = [stats['positive'] for _, stats in top_categories]
        negative = [stats['negative'] for _, stats in top_categories]
        neutral = [stats['neutral'] for _, stats in top_categories]
        
        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        
        plt.bar(x, positive, label='Положительные', color='green', alpha=0.7)
        plt.bar(x, neutral, bottom=positive, label='Нейтральные', color='gray', alpha=0.7)
        plt.bar(x, negative, bottom=[positive[i] + neutral[i] for i in range(len(categories))], 
                label='Отрицательные', color='red', alpha=0.7)
        
        plt.xlabel('Категории')
        plt.ylabel('Количество отзывов')
        plt.title('Распределение тональности по категориям')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        full_path = f"{self.output_dir}/{filename}"
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"   ✅ Сохранено: {full_path}")

    def plot_categories_summary(self, categories_stats, top_n=15):
        """
        Визуализация сводки по категориям
        """
        # Подготовка данных
        categories = []
        total_reviews = []
        positive_pct = []
        negative_pct = []
        
        sorted_cats = sorted(
            categories_stats.items(),
            key=lambda x: x[1]['total_reviews'],
            reverse=True
        )[:top_n]
        
        for cat, stats in sorted_cats:
            # Переводим название категории
            translated_cat = self.analyzer.translate_category(cat)
            categories.append(translated_cat[:20] + '...' if len(translated_cat) > 20 else translated_cat)
            total_reviews.append(stats['total_reviews'])
            pos = stats['sentiment']['positive'] / stats['total_reviews'] * 100
            neg = stats['sentiment']['negative'] / stats['total_reviews'] * 100
            positive_pct.append(pos)
            negative_pct.append(neg)
        
        # График 1: Количество отзывов по категориям
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(categories)), total_reviews, color='skyblue')
        plt.title(f'Топ-{top_n} категорий по количеству отзывов', fontsize=14)
        plt.xlabel('Категории')
        plt.ylabel('Количество отзывов')
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        
        # Добавляем значения
        for i, (bar, val) in enumerate(zip(bars, total_reviews)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_reviews)*0.01,
                    f'{val}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/categories_popularity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: categories_popularity.png")
        
        # График 2: Соотношение позитива/негатива
        plt.figure(figsize=(12, 6))
        x = range(len(categories))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], positive_pct, width, label='Позитив', color='green', alpha=0.7)
        plt.bar([i + width/2 for i in x], negative_pct, width, label='Негатив', color='red', alpha=0.7)
        
        plt.title(f'Соотношение позитива/негатива по категориям', fontsize=14)
        plt.xlabel('Категории')
        plt.ylabel('Процент отзывов')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/categories_sentiment.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: categories_sentiment.png")

    def create_category_wordclouds(self, results, category_stats=None, words_per_cloud=40):
        """
        Создание облаков слов для категорий.

        Parameters:
        results (dict): Результаты анализа (должен содержать category_frequencies)
        category_stats (dict, optional): Статистика по категориям для сортировки
        words_per_cloud (int): Сколько слов в облаке
        """
        if not results['has_category']:
            print("❌ Нет данных по категориям")
            return

        print("\n" + "=" * 60)
        print("☁️  СОЗДАНИЕ ОБЛАКОВ СЛОВ ДЛЯ ВСЕХ КАТЕГОРИЙ")
        print("=" * 60)

        category_freqs = results['category_frequencies']

        # Получаем ВСЕ категории, исключая unknown
        all_categories = [cat for cat in category_freqs.keys() if cat != 'unknown']

        if not all_categories:
            print("❌ Нет категорий для обработки")
            return

        print(f"📊 Всего категорий для обработки: {len(all_categories)}")
        
        # Показываем переведенные названия категорий
        translated_categories = [self.analyzer.translate_category(cat) for cat in all_categories]
        print(f"📊 Категории: {', '.join(translated_categories)}")

        created_count = 0
        for category in all_categories:
            translated_category = self.analyzer.translate_category(category)
            print(f"\n📊 Создание облаков для категории: {translated_category}")

            # Положительные слова
            if category in category_freqs and category_freqs[category]['positive']:
                pos_words = dict(category_freqs[category]['positive'].most_common(words_per_cloud))
                if pos_words:
                    try:
                        wordcloud = WordCloud(
                            width=1000,
                            height=500,
                            background_color='white',
                            max_words=words_per_cloud,
                            colormap='Greens',
                            random_state=42,
                            contour_width=1,
                            contour_color='darkgreen'
                        ).generate_from_frequencies(pos_words)

                        plt.figure(figsize=(12, 6))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title(f'Положительные отзывы: {translated_category}', fontsize=14, pad=20)
                        plt.tight_layout()

                        # Очищаем название категории для имени файла
                        clean_category = category.lower().replace(' ', '_').replace('-', '_')
                        filename = f'{self.output_dir}/positive_{clean_category}.png'
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"   ✅ Положительные: positive_{clean_category}.png")
                        created_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при создании положительного облака: {e}")

            # Отрицательные слова
            if category in category_freqs and category_freqs[category]['negative']:
                neg_words = dict(category_freqs[category]['negative'].most_common(words_per_cloud))
                if neg_words:
                    try:
                        wordcloud = WordCloud(
                            width=1000,
                            height=500,
                            background_color='white',
                            max_words=words_per_cloud,
                            colormap='Reds',
                            random_state=42,
                            contour_width=1,
                            contour_color='darkred'
                        ).generate_from_frequencies(neg_words)

                        plt.figure(figsize=(12, 6))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title(f'Отрицательные отзывы: {translated_category}', fontsize=14, pad=20)
                        plt.tight_layout()

                        clean_category = category.lower().replace(' ', '_').replace('-', '_')
                        filename = f'{self.output_dir}/negative_{clean_category}.png'
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"   ✅ Отрицательные: negative_{clean_category}.png")
                        created_count += 1
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при создании отрицательного облака: {e}")

        print(f"\n✅ Всего создано облаков: {created_count}")

    def create_category_summary_chart(self, results, top_n=20):
        """
        Создание сводного графика с топ-словами по категориям.

        Parameters:
        results (dict): Результаты анализа
        top_n (int): Количество категорий для отображения
        """
        if not results['has_category']:
            print("❌ Нет данных по категориям")
            return

        print("\n📊 СОЗДАНИЕ СВОДНОГО ГРАФИКА ПО КАТЕГОРИЯМ")

        category_freqs = results['category_frequencies']

        # Получаем все категории, исключая unknown
        all_categories = []
        for cat in category_freqs.keys():
            if cat != 'unknown':
                all_categories.append(cat)

        if not all_categories:
            print("❌ Нет категорий для отображения")
            return

        # Берем первые top_n категорий (или все, если их меньше)
        categories_to_show = all_categories[:min(top_n, len(all_categories))]
        
        # Переводим названия категорий
        translated_categories = [self.analyzer.translate_category(cat) for cat in categories_to_show]

        print(f"   Отображаю категории: {', '.join(translated_categories)}")

        # Создаем фигуру с подграфиками
        n_categories = len(categories_to_show)
        fig, axes = plt.subplots(n_categories, 2, figsize=(14, n_categories * 2.5))
        fig.suptitle('Топ-слова по категориям', fontsize=16, y=1)

        # Обрабатываем случай с одной категорией
        if n_categories == 1:
            # Преобразуем axes в двумерный массив для единообразия
            axes = axes.reshape(1, 2)

        for i, (category, translated_category) in enumerate(zip(categories_to_show, translated_categories)):
            # Положительные слова
            if category in category_freqs:
                pos_words_dict = category_freqs[category]['positive']
                if pos_words_dict:
                    # Берем топ-7 слов
                    top_pos = pos_words_dict.most_common(7)
                    if top_pos:
                        words = [w for w, _ in top_pos]
                        counts = [c for _, c in top_pos]

                        axes[i, 0].barh(words, counts, color='green', alpha=0.7)
                        axes[i, 0].set_title(f'{translated_category} - Положительные', fontsize=10)
                        axes[i, 0].invert_yaxis()
                        axes[i, 0].set_xlabel('Частота')

            # Отрицательные слова
            if category in category_freqs:
                neg_words_dict = category_freqs[category]['negative']
                if neg_words_dict:
                    # Берем топ-7 слов
                    top_neg = neg_words_dict.most_common(7)
                    if top_neg:
                        words = [w for w, _ in top_neg]
                        counts = [c for _, c in top_neg]

                        axes[i, 1].barh(words, counts, color='red', alpha=0.7)
                        axes[i, 1].set_title(f'{translated_category} - Отрицательные', fontsize=10)
                        axes[i, 1].invert_yaxis()
                        axes[i, 1].set_xlabel('Частота')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/category_words_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: category_words_summary.png")

    # ВИЗУАЛИЗАЦИЯ ДЛИНЫ ОТЗЫВОВ
    def plot_text_length_analysis(self, df, categories_stats=None, filename='text_length_analysis.png'):
        """
        Визуализация анализа длины отзывов.

        Parameters:
        df (DataFrame): Данные с отзывами
        categories_stats (dict): Статистика по категориям (опционально)
        filename (str): Имя файла для сохранения
        """
        print("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ ДЛИНЫ ОТЗЫВОВ")

        # Создаем копию DataFrame и добавляем колонку sentiment, если её нет
        df_copy = df.copy()

        # Если колонки sentiment нет, создаем её на основе рейтинга
        if 'sentiment' not in df_copy.columns:
            print("   ℹ️ Колонка 'sentiment' не найдена, создаем на основе рейтинга")
            df_copy['sentiment'] = 'neutral'
            df_copy.loc[df_copy['rating'] >= 4, 'sentiment'] = 'positive'
            df_copy.loc[df_copy['rating'] <= 2, 'sentiment'] = 'negative'

        # Создаем фигуру с несколькими подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Анализ длины отзывов', fontsize=16, y=0.98)

        # 1. Распределение длины отзывов (гистограмма)
        ax1 = axes[0, 0]
        text_lengths = df_copy['text'].dropna().astype(str).str.len()

        if len(text_lengths) > 0:
            # Ограничиваем для читаемости (берем 99-й перцентиль)
            max_len = text_lengths.quantile(0.99)
            filtered_lengths = text_lengths[text_lengths <= max_len]

            ax1.hist(filtered_lengths, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
            ax1.axvline(text_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Средняя: {text_lengths.mean():.0f} симв.')
            ax1.axvline(text_lengths.median(), color='green', linestyle='--', linewidth=2, label=f'Медиана: {text_lengths.median():.0f} симв.')
            ax1.set_xlabel('Длина отзыва (символы)')
            ax1.set_ylabel('Количество отзывов')
            ax1.set_title('Распределение длины отзывов')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax1.transAxes)

        # 2. Длина отзывов по тональности (boxplot)
        ax2 = axes[0, 1]

        # Подготавливаем данные по тональности
        sentiment_data = []
        sentiment_labels = []

        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_df = df_copy[df_copy['sentiment'] == sentiment]
            lengths = sentiment_df['text'].dropna().astype(str).str.len()
            if len(lengths) > 0:
                sentiment_data.append(lengths)
                sentiment_labels.append(f'{sentiment}\n({len(sentiment_df)} отзывов)')

        if sentiment_data:
            bp = ax2.boxplot(sentiment_data, labels=sentiment_labels, patch_artist=True)

            # Цвета для боксплотов
            colors = ['green', 'gray', 'red']
            for patch, color in zip(bp['boxes'], colors[:len(sentiment_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax2.set_ylabel('Длина отзыва (символы)')
            ax2.set_title('Длина отзывов по тональности')
            ax2.grid(axis='y', alpha=0.3)

            # Добавляем средние значения
            for i, (data, label) in enumerate(zip(sentiment_data, sentiment_labels), 1):
                mean_val = data.mean()
                ax2.text(i, mean_val, f'ср: {mean_val:.0f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax2.transAxes)

        # 3. Доля заполненных полей (столбчатая диаграмма)
        ax3 = axes[1, 0]

        text_filled = df_copy['text'].notna().sum() / len(df_copy) * 100
        plus_filled = df_copy['plus'].notna().sum() / len(df_copy) * 100
        minus_filled = df_copy['minus'].notna().sum() / len(df_copy) * 100

        fields = ['text', 'plus', 'minus']
        filled_pct = [text_filled, plus_filled, minus_filled]
        colors = ['lightgreen', 'gold', 'lightcoral']

        bars = ax3.bar(fields, filled_pct, color=colors, edgecolor='black')
        ax3.set_ylabel('Заполненность (%)')
        ax3.set_title('Доля заполненных полей в отзывах')
        ax3.set_ylim(0, 100)

        for bar, pct in zip(bars, filled_pct):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Средняя длина по категориям (топ-15)
        ax4 = axes[1, 1]

        if categories_stats:
            # Собираем данные по категориям
            categories_data = []

            for cat, stats in categories_stats.items():
                if stats['total_reviews'] >= 10:
                    categories_data.append({
                        'name': self.analyzer.translate_category(cat),
                        'avg_length': stats['text_length']['avg'],
                        'total_reviews': stats['total_reviews']
                    })

            if categories_data:
                # Сортируем по средней длине отзыва (по убыванию)
                categories_data.sort(key=lambda x: x['avg_length'], reverse=True)

                # Берем топ-15
                categories_data = categories_data[:15]

                categories = [item['name'][:25] for item in categories_data]  # Ограничиваем длину названия
                avg_lengths = [item['avg_length'] for item in categories_data]
                total_reviews = [item['total_reviews'] for item in categories_data]

                # Горизонтальная гистограмма для лучшей читаемости
                y_pos = range(len(categories))
                bars = ax4.barh(y_pos, avg_lengths, color='skyblue', edgecolor='navy')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(categories)
                ax4.set_xlabel('Средняя длина отзыва (символы)')
                ax4.set_title('Средняя длина отзыва по категориям (сортировка по длине)')
                ax4.invert_yaxis()

                # Добавляем значения и подписи внутри/снаружи
                for i, (bar, val, reviews) in enumerate(zip(bars, avg_lengths, total_reviews)):
                    # Если столбец достаточно широк, размещаем подпись внутри
                    if val > max(avg_lengths) * 0.15:
                        ax4.text(val - 5, bar.get_y() + bar.get_height()/2,
                                f'{val:.0f} ({reviews} отз.)', 
                                va='center', ha='right', fontsize=8, color='white', fontweight='bold')
                    else:
                        # Если столбец узкий, размещаем подпись снаружи справа
                        ax4.text(val + 5, bar.get_y() + bar.get_height()/2,
                                f'{val:.0f} ({reviews} отз.)', 
                                va='center', ha='left', fontsize=8, color='navy')

                ax4.grid(axis='x', alpha=0.3)

                # Добавляем вертикальную линию среднего значения
                all_avg = sum(avg_lengths) / len(avg_lengths)
                ax4.axvline(all_avg, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Общее среднее: {all_avg:.0f}')
                ax4.legend(loc='lower right', fontsize=8)

                # Добавляем подпись с информацией о сортировке
                ax4.text(0.98, 0.02, '↗️ Сортировка: от длинных к коротким', 
                        transform=ax4.transAxes, ha='right', va='bottom', 
                        fontsize=8, style='italic', color='gray')

            else:
                ax4.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Нет данных по категориям', ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: {filename}")

    # ВИЗУАЛИЗАЦИЯ ДЛИНЫ ПО ТОНАЛЬНОСТИ ПО КАТЕГОРИЯМ
    def plot_sentiment_length_by_category(self, categories_stats, filename='sentiment_length_by_category.png'):
        """
        Визуализация длины положительных и отрицательных отзывов по категориям.
        Помогает понять, какие категории вызывают более эмоциональные отклики.
        
        Parameters:
        categories_stats (dict): Статистика по категориям
        filename (str): Имя файла для сохранения
        """
        print("\n📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИИ: Длина отзывов по тональности в разрезе категорий")
        
        if not categories_stats:
            print("❌ Нет данных по категориям")
            return
        
        # Собираем данные
        categories = []
        positive_lengths = []
        negative_lengths = []
        neutral_lengths = []
        
        # Сортируем по популярности
        sorted_cats = sorted(
            categories_stats.items(),
            key=lambda x: x[1]['total_reviews'],
            reverse=True
        )[:12]  # Топ-12 категорий для читаемости
        
        for cat, stats in sorted_cats:
            if stats['total_reviews'] >= 10:
                categories.append(self.analyzer.translate_category(cat))
                positive_lengths.append(stats.get('positive_length', {}).get('avg', 0))
                negative_lengths.append(stats.get('negative_length', {}).get('avg', 0))
                neutral_lengths.append(stats.get('neutral_length', {}).get('avg', 0))
        
        if not categories:
            print("❌ Нет данных для визуализации")
            return
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = range(len(categories))
        width = 0.25
        
        bars1 = ax.bar([i - width for i in x], positive_lengths, width, 
                       label='Положительные', color='green', alpha=0.7)
        bars2 = ax.bar(x, neutral_lengths, width, 
                       label='Нейтральные', color='gray', alpha=0.7)
        bars3 = ax.bar([i + width for i in x], negative_lengths, width, 
                       label='Отрицательные', color='red', alpha=0.7)
        
        ax.set_xlabel('Категории')
        ax.set_ylabel('Средняя длина отзыва (символы)')
        ax.set_title('Средняя длина отзывов по тональности в разрезе категорий')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Добавляем значения на столбцы
        def add_values(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        add_values(bars1)
        add_values(bars2)
        add_values(bars3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: {filename}")
        
        # Вывод аналитики в консоль
        print("\n📊 АНАЛИТИКА ПО ДЛИНЕ ОТЗЫВОВ:")
        print("-" * 60)
        
        # Категории с самой большой разницей между позитивом и негативом
        diff_data = []
        for i, cat in enumerate(categories):
            if positive_lengths[i] > 0 and negative_lengths[i] > 0:
                diff = positive_lengths[i] - negative_lengths[i]
                diff_data.append((cat, diff, positive_lengths[i], negative_lengths[i]))
        
        diff_data.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 Категории с наибольшей разницей в длине отзывов:")
        for cat, diff, pos_len, neg_len in diff_data[:5]:
            print(f"   {cat}: позитив {pos_len:.0f} vs негатив {neg_len:.0f} (разница {diff:.0f} симв.)")
        
        print("\n⚠️ Категории, где негативные отзывы длиннее позитивных:")
        for cat, diff, pos_len, neg_len in diff_data:
            if diff < 0:
                print(f"   {cat}: негатив {neg_len:.0f} > позитив {pos_len:.0f} (на {abs(diff):.0f} симв.)")
        
        # Категории с самыми длинными отзывами
        length_data = [(cat, pos_len, neg_len) for cat, pos_len, neg_len 
                       in zip(categories, positive_lengths, negative_lengths)]
        length_data.sort(key=lambda x: max(x[1], x[2]), reverse=True)
        
        print("\n📝 Топ-5 категорий с самыми длинными отзывами:")
        for cat, pos_len, neg_len in length_data[:5]:
            max_len = max(pos_len, neg_len)
            print(f"   {cat}: {max_len:.0f} симв. (позитив: {pos_len:.0f}, негатив: {neg_len:.0f})")