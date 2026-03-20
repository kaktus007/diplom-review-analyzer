#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Главный скрипт для запуска анализа отзывов.
Дипломный проект.
"""

import os
import sys
import argparse
from collections import Counter

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import DiplomReviewAnalyzer
from visualization import ReviewVisualizer
import utils


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Анализ отзывов из CSV файла')
    parser.add_argument('--input', '-i', type=str, default='data/raw/reviews.csv',
                       help='Путь к входному CSV файлу')
    parser.add_argument('--encoding', '-e', type=str, default=None,
                       help='Кодировка файла')
    parser.add_argument('--stopwords', '-s', type=str, default='custom_stopwords.txt',
                       help='Файл с кастомными стоп-словами')
    parser.add_argument('--no-progress', action='store_true',
                       help='Отключить прогресс-бар')
    parser.add_argument('--suggest', action='store_true',
                       help='Предложить стоп-слова на основе анализа')
    return parser.parse_args()


def interactive_stopwords_menu(analyzer, results):
    """Интерактивное меню для управления стоп-словами"""
    while True:
        print("\n" + "=" * 60)
        print("🛑 УПРАВЛЕНИЕ СТОП-СЛОВАМИ")
        print("=" * 60)
        print("1. Показать текущие кастомные стоп-слова")
        print("2. Добавить стоп-слово")
        print("3. Добавить несколько стоп-слов")
        print("4. Удалить стоп-слово")
        print("5. Предложить стоп-слова на основе анализа")
        print("6. Сохранить стоп-слова в файл")
        print("7. Загрузить стоп-слова из файла")
        print("8. Выполнить анализ с новыми стоп-словами")
        print("0. Выйти в главное меню")
        
        choice = input("\nВыберите действие (0-8): ").strip()
        
        if choice == '1':
            print("\n📋 Текущие кастомные стоп-слова:")
            if analyzer.custom_stopwords:
                for word in sorted(analyzer.custom_stopwords):
                    print(f"  • {word}")
                print(f"\nВсего: {len(analyzer.custom_stopwords)} слов")
            else:
                print("  Нет кастомных стоп-слов")
        
        elif choice == '2':
            word = input("Введите стоп-слово: ").strip().lower()
            if word:
                analyzer.add_custom_stopwords([word])
        
        elif choice == '3':
            words = input("Введите стоп-слова через запятую: ").strip().lower()
            if words:
                word_list = [w.strip() for w in words.split(',') if w.strip()]
                analyzer.add_custom_stopwords(word_list)
        
        elif choice == '4':
            word = input("Введите стоп-слово для удаления: ").strip().lower()
            if word:
                analyzer.remove_custom_stopwords([word])
        
        elif choice == '5':
            print("\n🔍 Анализ частотных слов...")
            suggestions = analyzer.suggest_stopwords_from_results(results, top_n=50, min_freq=1000)
            if suggestions:
                add = input("\nДобавить предложенные слова? (y/n): ").strip().lower()
                if add == 'y':
                    analyzer.add_custom_stopwords(suggestions)
        
        elif choice == '6':
            filename = input("Имя файла для сохранения (по умолчанию custom_stopwords.txt): ").strip()
            if not filename:
                filename = 'custom_stopwords.txt'
            analyzer.save_stopwords_to_file(filename)
        
        elif choice == '7':
            filename = input("Имя файла для загрузки: ").strip()
            if filename and os.path.exists(filename):
                new_analyzer = DiplomReviewAnalyzer(custom_stopwords_file=filename)
                analyzer.custom_stopwords = new_analyzer.custom_stopwords
                analyzer.stopwords.update(analyzer.custom_stopwords)
                print(f"✅ Загружено {len(analyzer.custom_stopwords)} стоп-слов")
            else:
                print("❌ Файл не найден")
        
        elif choice == '8':
            print("\n🔄 Запуск анализа с новыми стоп-словами...")
            return True
        
        elif choice == '0':
            return False
        
        else:
            print("❌ Неверный выбор")


def main():
    """Основная функция"""
    print("=" * 60)
    print("ДИПЛОМНЫЙ ПРОЕКТ: АНАЛИЗ ОТЗЫВОВ")
    print("=" * 60)
    
    # Парсим аргументы
    args = parse_arguments()
    input_file = args.input
    encoding = args.encoding
    stopwords_file = args.stopwords
    use_progress = not args.no_progress
    
    # Проверяем существование файла
    if not os.path.exists(input_file):
        print(f"❌ Файл {input_file} не найден!")
        return
    
    # Создаем папки для результатов
    os.makedirs('output/images', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)
    os.makedirs('output/csv', exist_ok=True)
    
    # Создаем анализатор с кастомными стоп-словами
    analyzer = DiplomReviewAnalyzer(
        use_tqdm=use_progress,
        custom_stopwords_file=stopwords_file if os.path.exists(stopwords_file) else None
    )
    visualizer = ReviewVisualizer(output_dir='output/images')
    
    # Основной цикл анализа
    while True:
        # 1. Загружаем данные
        print(f"\n📂 ЗАГРУЗКА ДАННЫХ ИЗ ФАЙЛА {input_file}")
        df = analyzer.load_from_csv(input_file, encoding)
        
        if df is None:
            return
        
        # 2. Информация о данных
        print("\n🔍 ИНФОРМАЦИЯ О ДАННЫХ:")
        print(f"Колонки: {list(df.columns)}")
        print(f"Строк: {len(df)}")
        
        # 3. Анализ отзывов
        results = analyzer.analyze_reviews(df)
        
        if results is None:
            print("❌ Ошибка при анализе отзывов")
            return
        
        # 4. Анализ категорий (если есть колонка category)
        if 'category' in df.columns:
            categories_stats = analyzer.analyze_categories(df, min_reviews=5)
            
            if categories_stats:
                # Сохраняем отчет
                utils.save_categories_report(categories_stats, results['df'], 'output/reports/categories_report.txt')
                
                # Визуализируем
                visualizer.plot_categories_summary(categories_stats, top_n=15)
                
                # На основе анализа категорий предлагаем стоп-слова
                # На основе анализа категорий предлагаем стоп-слова
                print("\n" + "=" * 60)
                print("💡 РЕКОМЕНДАЦИИ ПО СТОП-СЛОВАМ НА ОСНОВЕ КАТЕГОРИЙ")
                print("=" * 60)
                
                # Собираем общие слова из всех категорий
                all_category_words = []
                for cat, stats in categories_stats.items():
                    if stats['total_reviews'] >= 10:
                        for word, _ in stats['top_plus_words'] + stats['top_minus_words']:
                            all_category_words.append(word)
                
                # Самые частотные слова по категориям
                if all_category_words:
                    common_words = Counter(all_category_words).most_common(30)
                    print("\n📊 ТОП-30 ЛЕММАТИЗИРОВАННЫХ СЛОВ, ЧАСТО ВСТРЕЧАЮЩИХСЯ В РАЗНЫХ КАТЕГОРИЯХ:")
                    print("=" * 70)
                    print(f"{'Слово':<20} | {'Категорий':<10} | {'Статус':<15}")
                    print("-" * 50)
                    
                    for word, count in common_words[:20]:
                        if word in analyzer.stopwords:
                            status = "✅ УЖЕ В СТОП-СЛОВАХ"
                        else:
                            status = "🆕 МОЖНО ДОБАВИТЬ"
                        print(f"{word:<20} | {count:<10} | {status}")
                    
                    # Предлагаем новые стоп-слова
                    new_stopwords = [word for word, count in common_words[:15] 
                                     if word not in analyzer.stopwords and count >= 3]
                    
                    if new_stopwords:
                        print("\n🆕 НОВЫЕ СТОП-СЛОВА ДЛЯ ДОБАВЛЕНИЯ:")
                        print(", ".join(new_stopwords))
                    else:
                        print("\n✅ Нет новых слов для добавления (все частотные слова уже в стоп-списке)")
        
        # 5. Сохраняем результаты
        utils.save_results_to_csv(results, 'output/csv/analyzed_reviews.csv')
        utils.save_report(results, 'output/reports/analysis_report.txt')
        
        # 6. Показываем топ слова
        print("\n" + "=" * 60)
        print("ТОП-20 СЛОВ ПО КАЖДОЙ КАТЕГОРИИ")
        print("=" * 60)
        
        utils.print_top_words(results['frequencies']['positive'], "ПОЛОЖИТЕЛЬНЫЕ СЛОВА", 20)
        utils.print_top_words(results['frequencies']['negative'], "ОТРИЦАТЕЛЬНЫЕ СЛОВА", 20)
        utils.print_top_words(results['frequencies']['neutral'], "НЕЙТРАЛЬНЫЕ СЛОВА", 20)
        
        # 7. Предложить стоп-слова, если запрошено
        if args.suggest:
            analyzer.suggest_stopwords_from_results(results, top_n=50, min_freq=1000)
            args.suggest = False  # только один раз
        
        # 8. Интерактивное меню
        if not interactive_stopwords_menu(analyzer, results):
            break
        
        print("\n" + "=" * 60)
        print("🔄 ПОВТОРНЫЙ АНАЛИЗ С НОВЫМИ СТОП-СЛОВАМИ")
        print("=" * 60)
    
    # 9. Финальная визуализация
    print("\n" + "=" * 60)
    print("🖼️  СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 60)

    print("\nСоздание общих облаков слов...")
    visualizer.create_wordcloud(
        results['frequencies']['positive'], 
        "Положительные отзывы (все категории)", 
        "positive_cloud.png",
        color='Greens'
    )

    visualizer.create_wordcloud(
        results['frequencies']['negative'], 
        "Отрицательные отзывы (все категории)", 
        "negative_cloud.png",
        color='Reds'
    )

    visualizer.create_wordcloud(
        results['frequencies']['neutral'], 
        "Нейтральные отзывы (все категории)", 
        "neutral_cloud.png",
        color='Purples'
    )

    print("\nСоздание облаков слов по категориям...")
    visualizer.create_category_wordclouds(
        results,  
        words_per_cloud=40
    )

    # Визуализация длины отзывов
    print("\n📊 Визуализация длины отзывов...")
    visualizer.plot_text_length_analysis(
        results['df'], 
        categories_stats if 'categories_stats' in locals() else None,
        'text_length_analysis.png'
    )

    if 'categories_stats' in locals() and categories_stats:
        visualizer.plot_sentiment_length_by_category(
            categories_stats,
            'sentiment_length_by_category.png'
        )

    # Сохраняем лемматизированные слова в файл
    print("\n📝 Сохранение лемматизированных слов в файл...")
    utils.save_lemmatized_words(results, 'output/reports/lemmatized_words.txt')

    print("\nСоздание сводных графиков...")
    visualizer.plot_rating_distribution(results['df'])
    visualizer.plot_field_usage(results['df'])

    if results['has_category']:
        visualizer.create_category_chart(results)
        visualizer.create_category_summary_chart(results, top_n=20)


if __name__ == "__main__":
    main()