"""
Вспомогательные функции для проекта.
"""

import json
import pandas as pd
from collections import Counter


def save_results_to_csv(results, filename='output/csv/analyzed_reviews.csv'):
    """Сохранение обработанных данных"""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    df_output = results['df'].copy()
    df_output['has_text'] = df_output['text'].notna()
    df_output['has_plus'] = df_output['plus'].notna()
    df_output['has_minus'] = df_output['minus'].notna()
    
    df_output.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"   ✅ Сохранено: {filename}")


def save_report(results, filename='output/reports/analysis_report.txt'):
    """Сохранение отчета"""
    import os
    from analyzer import DiplomReviewAnalyzer
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Создаем временный экземпляр анализатора для перевода
    analyzer = DiplomReviewAnalyzer(use_tqdm=False)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("ДИПЛОМНЫЙ ПРОЕКТ: АНАЛИЗ ОТЗЫВОВ\n")
        f.write("=" * 50 + "\n\n")
        
        # Общая статистика
        f.write("1. ОБЩАЯ СТАТИСТИКА\n")
        f.write("-" * 30 + "\n")
        f.write(f"Всего отзывов: {len(results['df'])}\n")
        f.write(f"Положительных: {results['stats']['sentiment_counts'].get('positive', 0)}\n")
        f.write(f"Отрицательных: {results['stats']['sentiment_counts'].get('negative', 0)}\n")
        f.write(f"Нейтральных: {results['stats']['sentiment_counts'].get('neutral', 0)}\n\n")
        
        # Топ-20 слов
        f.write("4. ТОП-20 СЛОВ ПО КАТЕГОРИЯМ\n")
        f.write("-" * 30 + "\n")
        
        for sentiment in ['positive', 'negative', 'neutral']:
            f.write(f"\n{sentiment.upper()}:\n")
            freq = results['frequencies'][sentiment]
            for i, (word, count) in enumerate(freq.most_common(20), 1):
                f.write(f"  {i:2d}. {word:<20} {count}\n")
        
        # Анализ по категориям
        if results['has_category']:
            f.write("\n\n5. АНАЛИЗ ПО КАТЕГОРИЯМ\n")
            f.write("=" * 30 + "\n")
            
            category_stats = results['stats']['category_stats']
            
            # Точки роста
            f.write("\nТОЧКИ РОСТА (высокий негатив):\n")
            for category, stats in sorted(category_stats.items(), 
                                        key=lambda x: x[1]['negative']/x[1]['total'] if x[1]['total']>0 else 0,
                                        reverse=True)[:10]:
                if category != 'unknown' and stats['total'] >= 10:
                    neg_share = stats['negative'] / stats['total'] * 100
                    translated_category = analyzer.translate_category(category)
                    f.write(f"\n{translated_category}: {neg_share:.1f}% негатива")
    
    print(f"   ✅ Сохранено: {filename}")


def print_top_words(freq_dict, title, n=20):
    """Вывод топ слов в консоль"""
    if not freq_dict:
        print(f"\n❌ Нет данных для {title}")
        return
    
    print(f"\n=== {title} (топ-{n}) ===")
    print(f"{'Слово':<20} | {'Частота':<10}")
    print("-" * 32)
    
    for word, count in freq_dict.most_common(n):
        print(f"{word:<20} | {count:<10}")


def save_categories_report(categories_stats, df_filtered, filename='output/reports/categories_report.txt'):
    """
    Сохранение отчета по категориям в текстовый файл
    
    Parameters:
    categories_stats (dict): Статистика по категориям
    df_filtered (DataFrame): Отфильтрованный DataFrame с русскоязычными отзывами
    filename (str): Путь для сохранения файла
    """
    import os
    from analyzer import DiplomReviewAnalyzer
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Создаем временный экземпляр анализатора для перевода
    analyzer = DiplomReviewAnalyzer(use_tqdm=False)
    
    # Подсчитываем общее количество русскоязычных отзывов в категориях
    # (исключаем отзывы без категории, если они есть)
    if 'category' in df_filtered.columns:
        total_russian_reviews = len(df_filtered[df_filtered['category'] != 'unknown'])
    else:
        total_russian_reviews = len(df_filtered)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЕТ ПО КАТЕГОРИЯМ ТОВАРОВ\n")
        f.write("=" * 60 + "\n\n")
        
        # Общая статистика
        total_categories = len(categories_stats)
        
        f.write(f"Всего категорий: {total_categories}\n")
        f.write(f"Всего отзывов в анализируемых категориях (русскоязычные): {total_russian_reviews}\n")
        f.write(f"   (фильтрация по языку: только русские отзывы)\n\n")
        
        # ДЕТАЛЬНАЯ СТАТИСТИКА ПО КАЖДОЙ КАТЕГОРИИ
        f.write("\n" + "=" * 60 + "\n")
        f.write("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ПО КАТЕГОРИЯМ\n")
        f.write("=" * 60 + "\n")
        
        # Сортируем по количеству отзывов
        sorted_categories = sorted(
            categories_stats.items(),
            key=lambda x: x[1]['total_reviews'],
            reverse=True
        )
        
        for i, (category, stats) in enumerate(sorted_categories, 1):
            # Пропускаем категорию unknown, если она есть
            if category == 'unknown':
                continue
                
            translated_category = analyzer.translate_category(category)
            f.write(f"\n{i:2d}. 📍 {translated_category.upper()}\n")
            f.write(f"   📊 Всего отзывов: {stats['total_reviews']}\n")
            f.write(f"   ⭐ Средний рейтинг: {stats['avg_rating']}\n")
            
            # Длина отзывов
            f.write(f"   📝 Длина отзывов:\n")
            f.write(f"      text: мин {stats['text_length']['min']:.0f} | ср {stats['text_length']['avg']:.0f} | макс {stats['text_length']['max']:.0f} симв.\n")
            f.write(f"      plus: мин {stats['plus_length']['min']:.0f} | ср {stats['plus_length']['avg']:.0f} | макс {stats['plus_length']['max']:.0f} симв.\n")
            f.write(f"      minus: мин {stats['minus_length']['min']:.0f} | ср {stats['minus_length']['avg']:.0f} | макс {stats['minus_length']['max']:.0f} симв.\n")
            
            # Тональность
            pos = stats['sentiment_percent']['positive']
            neu = stats['sentiment_percent']['neutral']
            neg = stats['sentiment_percent']['negative']
            f.write(f"   📈 Тональность: 🟢 {pos}% | ⚪ {neu}% | 🔴 {neg}%\n")
            
            # Заполненность полей
            text_pct = stats['fields_percent']['text']
            plus_pct = stats['fields_percent']['plus']
            minus_pct = stats['fields_percent']['minus']
            f.write(f"   📊 Заполненность: text {text_pct:.0f}% | plus {plus_pct:.0f}% | minus {minus_pct:.0f}%\n")
            
            # Топ слова из plus
            if stats['top_plus_words']:
                plus_words = ', '.join([f"{w}({c})" for w, c in stats['top_plus_words'][:5]])
                f.write(f"   👍 Часто в PLUS: {plus_words}\n")
            else:
                f.write(f"   👍 Часто в PLUS: нет данных\n")
            
            # Топ слова из minus
            if stats['top_minus_words']:
                minus_words = ', '.join([f"{w}({c})" for w, c in stats['top_minus_words'][:5]])
                f.write(f"   👎 Часто в MINUS: {minus_words}\n")
            else:
                f.write(f"   👎 Часто в MINUS: нет данных\n")
        
        # Категории с высоким негативом (точки роста)
        f.write("\n" + "=" * 60 + "\n")
        f.write("🎯 ТОЧКИ РОСТА (категории с высоким процентом негатива)\n")
        f.write("=" * 60 + "\n")
        
        growth_categories = sorted(
            [(cat, stats) for cat, stats in categories_stats.items() 
             if stats['total_reviews'] >= 10 and cat != 'unknown'],
            key=lambda x: x[1]['sentiment']['negative'] / x[1]['total_reviews'],
            reverse=True
        )
        
        for i, (category, stats) in enumerate(growth_categories, 1):
            translated_category = analyzer.translate_category(category)
            neg_share = stats['sentiment']['negative'] / stats['total_reviews'] * 100
            pos_share = stats['sentiment']['positive'] / stats['total_reviews'] * 100
            f.write(f"\n{i:2d}. {translated_category}\n")
            f.write(f"     Всего: {stats['total_reviews']} отзывов\n")
            f.write(f"     🟢 Позитив: {stats['sentiment']['positive']} ({pos_share:.1f}%)\n")
            f.write(f"     🔴 Негатив: {stats['sentiment']['negative']} ({neg_share:.1f}%)\n")
            
            if stats['top_minus_words']:
                f.write("     Частые проблемы:\n")
                for word, count in stats['top_minus_words'][:7]:
                    f.write(f"       • {word}: {count}\n")
        
        # Самые популярные категории
        f.write("\n" + "=" * 60 + "\n")
        f.write("🔥 САМЫЕ ПОПУЛЯРНЫЕ КАТЕГОРИИ\n")
        f.write("=" * 60 + "\n")
        
        popular = sorted(
            [(cat, stats) for cat, stats in categories_stats.items() if cat != 'unknown'],
            key=lambda x: x[1]['total_reviews'],
            reverse=True
        )
        
        for i, (category, stats) in enumerate(popular, 1):
            translated_category = analyzer.translate_category(category)
            f.write(f"\n{i:2d}. {translated_category}\n")
            f.write(f"     Отзывов: {stats['total_reviews']}\n")
            f.write(f"     Средний рейтинг: {stats['avg_rating']}\n")
            f.write(f"     Тональность: 🟢 {stats['sentiment']['positive']} | ⚪ {stats['sentiment']['neutral']} | 🔴 {stats['sentiment']['negative']}\n")
            
            # Добавляем длину отзывов для популярных категорий
            f.write(f"     Длина отзывов: text {stats['text_length']['avg']:.0f} симв.\n")
    
    print(f"✅ Отчет по категориям сохранен в {filename}")


def save_lemmatized_words(results, filename='output/reports/lemmatized_words.txt'):
    """
    Сохранение всех лемматизированных слов по категориям в текстовый файл.
    
    Формат:
    1. НАЗВАНИЕ КАТЕГОРИИ
    слово1
    слово2
    ...
    
    Parameters:
    results (dict): Результаты анализа (должен содержать category_frequencies)
    filename (str): Путь для сохранения файла
    """
    import os
    from analyzer import DiplomReviewAnalyzer
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Создаем временный экземпляр анализатора для перевода
    analyzer = DiplomReviewAnalyzer(use_tqdm=False)
    
    if not results.get('has_category', False):
        print("❌ Нет данных по категориям для сохранения слов")
        return
    
    category_freqs = results.get('category_frequencies', {})
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ЛЕММАТИЗИРОВАННЫЕ СЛОВА ПО КАТЕГОРИЯМ\n")
        f.write("=" * 60 + "\n\n")
        
        # Счетчик для нумерации категорий
        category_counter = 1
        
        # Проходим по всем категориям
        for category, sentiments in category_freqs.items():
            if category == 'unknown':
                continue
            
            # Переводим название категории
            translated_category = analyzer.translate_category(category)
            f.write(f"{category_counter}. {translated_category.upper()}\n")
            f.write("-" * 40 + "\n")
            
            # Положительные слова
            if sentiments['positive']:
                f.write("\n[ПОЛОЖИТЕЛЬНЫЕ]\n")
                for word, count in sorted(sentiments['positive'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{word} ({count})\n")
            
            # Отрицательные слова
            if sentiments['negative']:
                f.write("\n[ОТРИЦАТЕЛЬНЫЕ]\n")
                for word, count in sorted(sentiments['negative'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{word} ({count})\n")
            
            # Нейтральные слова
            if sentiments['neutral']:
                f.write("\n[НЕЙТРАЛЬНЫЕ]\n")
                for word, count in sorted(sentiments['neutral'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{word} ({count})\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            category_counter += 1
        
        # Добавляем общую статистику
        all_words = results.get('all_words', {})
        if all_words:
            f.write("\n" + "=" * 60 + "\n")
            f.write("ВСЕ СЛОВА (ОБЩАЯ СТАТИСТИКА)\n")
            f.write("=" * 60 + "\n\n")
            for word, count in all_words.most_common(100):
                f.write(f"{word}: {count}\n")
    
    print(f"✅ Лемматизированные слова сохранены в {filename}")
    
    # Также создаем упрощенную версию без частот (только слова)
    simple_filename = filename.replace('.txt', '_simple.txt')
    with open(simple_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ЛЕММАТИЗИРОВАННЫЕ СЛОВА ПО КАТЕГОРИЯМ (ТОЛЬКО СЛОВА)\n")
        f.write("=" * 60 + "\n\n")
        
        for category, sentiments in category_freqs.items():
            if category == 'unknown':
                continue
            
            # Переводим название категории
            translated_category = analyzer.translate_category(category)
            f.write(f"\n{translated_category.upper()}\n")
            f.write("-" * 40 + "\n")
            
            all_category_words = set()
            
            for sentiment in ['positive', 'negative', 'neutral']:
                all_category_words.update(sentiments[sentiment].keys())
            
            for word in sorted(all_category_words):
                f.write(f"{word}\n")
    
    print(f"✅ Упрощенный список слов сохранен в {simple_filename}")
    
    return filename