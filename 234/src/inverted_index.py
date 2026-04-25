import os
import pickle
from datetime import date
from pyroaring import BitMap
from src.text_processor import TextProcessor

class LSMInvertedIndex:
    def __init__(self, data_dir: str = "./data", flush_threshold: int = 1000):
        self.processor = TextProcessor()
        self.memtable = {} 
        self.data_dir = data_dir
        self.flush_threshold = flush_threshold 
        self.doc_count = 0
        self.doc_metadata = {} 
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def add_document(self, doc_id: int, text: str, created_at: date = None, valid_from: date = None, valid_to: date = None):
        # Сохр метаданные документа
        self.doc_metadata[doc_id] = {
            'created_at': created_at,
            'valid_from': valid_from,
            'valid_to': valid_to
        }

        tokens = self.processor.process(text)
        
        for token in tokens:
            if token not in self.memtable:
                self.memtable[token] = BitMap()
            self.memtable[token].add(doc_id)
            
        self.doc_count += 1
        
        if self.doc_count >= self.flush_threshold:
            self.flush_to_disk()

    def _get_word_bitmap(self, word: str) -> BitMap:
        result_bitmap = BitMap()
        
        # Поиск в оперативной памяти
        if word in self.memtable:
            result_bitmap |= self.memtable[word] # Объединяем результаты (логическое ИЛИ)
            
        # Поиск на диске во всех сохраненных сегментах
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'rb') as f:
                    segment = pickle.load(f)
                    if word in segment:
                        result_bitmap |= segment[word]
                        
        return result_bitmap

    def search_boolean_and(self, query: str) -> BitMap:
        # Поиск документов по логическому И
        tokens = self.processor.process(query)
        if not tokens:
            return BitMap()

        # Получаем битмап первого слова
        result = self._get_word_bitmap(tokens[0])
        
        if not result:
            return BitMap()
        
        # Пересекаем с остальными словами запроса
        for word in tokens[1:]:
            word_bitmap = self._get_word_bitmap(word)
            result &= word_bitmap # Оставляем только те документы, где есть ВСЕ слова (И)
            
        return result
    
    def _get_vocabulary(self) -> set:
        vocab = set(self.memtable.keys())
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'rb') as f:
                    segment = pickle.load(f)
                    vocab.update(segment.keys())
                    
        return vocab

    def search_prefix(self, prefix: str) -> BitMap:
        # Префикс тоже нужно прогнать через стеммер (но без очистки от стоп-слов)
        processed_prefix = self.processor.stemmer.stem(prefix.lower())
        
        result_bitmap = BitMap()
        all_words = self._get_vocabulary()
        
        # Ищем все слова
        matched_words = [w for w in all_words if w.startswith(processed_prefix)]
        
        # Объед результаты для всех найденных слов
        for word in matched_words:
            result_bitmap |= self._get_word_bitmap(word)
            
        return result_bitmap

    def search_wildcard(self, wildcard_query: str, k: int = 3) -> BitMap:
        import re
        # (Стеммер тут не используем, так как куски слов могут застеммиться криво)
        query_lower = wildcard_query.lower()
        parts = query_lower.split('*')
        
        # Собираем k-граммы
        query_kgrams = set()
        for part in parts:
            if len(part) >= k:
                # Скользящее окно размером k
                for i in range(len(part) - k + 1):
                    query_kgrams.add(part[i:i+k])

        # Создаем регулярку для точной проверки в конце (заменяем * на .*)
        regex_pattern = '^' + query_lower.replace('*', '.*') + '$'
        regex = re.compile(regex_pattern)

        result_bitmap = BitMap()
        all_words = self._get_vocabulary()
        
        matched_words = []
        
        # Ищем подходящие слова в словаре
        for word in all_words:
            # Бьем слово из словаря на k-граммы
            word_kgrams = set(word[i:i+k] for i in range(len(word) - k + 1))
            
            # Быстрый фильтр: все ли k-граммы запроса есть в слове?
            if query_kgrams.issubset(word_kgrams):
                # Точный фильтр: совпадает ли с регуляркой?
                if regex.match(word):
                    matched_words.append(word)

        for word in matched_words:
            result_bitmap |= self._get_word_bitmap(word)
            
        return result_bitmap

    def flush_to_disk(self):
        if not self.memtable:
            return
        
        sorted_keys = sorted(self.memtable.keys())
        segment_data = {k: self.memtable[k] for k in sorted_keys}

        segment_name = f"segment_{len(os.listdir(self.data_dir))}.pkl"
        segment_path = os.path.join(self.data_dir, segment_name)
        
        # Сохр на диск
        with open(segment_path, 'wb') as f:
            pickle.dump(segment_data, f)
            
        print(f"[LSM] MemTable сброшена на диск: {segment_name}")
        
        # Очистка памяти
        self.memtable.clear()
        self.doc_count = 0

    def get_date_range_bitmap(self, start_date: date, end_date: date) -> BitMap:
        result = BitMap()
        for doc_id, meta in self.doc_metadata.items():
            dt = meta.get('created_at')
            if dt and start_date <= dt <= end_date:
                result.add(doc_id)
        return result

    def get_valid_in_range_bitmap(self, start_date: date, end_date: date) -> BitMap:
        result = BitMap()
        for doc_id, meta in self.doc_metadata.items():
            v_from = meta.get('valid_from')
            v_to = meta.get('valid_to')
            
            # Документ начал жить ДО конца интервала и умер ПОСЛЕ начала интервала (либо еще жив, то есть None)
            if v_from and v_from <= end_date:
                if v_to is None or v_to >= start_date:
                    result.add(doc_id)
        return result

    def get_appeared_in_range_bitmap(self, start_date: date, end_date: date) -> BitMap:
        result = BitMap()
        for doc_id, meta in self.doc_metadata.items():
            v_from = meta.get('valid_from')
            if v_from and start_date <= v_from <= end_date:
                result.add(doc_id)
        return result
