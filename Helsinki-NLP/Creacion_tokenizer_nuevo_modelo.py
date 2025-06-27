# vocabulary_expansion_flores_galician.py

import torch
from transformers import MarianTokenizer, MarianMTModel
from datasets import load_dataset
from collections import Counter, defaultdict
import re
import numpy as np
from typing import List, Tuple, Dict

def analyze_galician_corpus_flores(min_freq=3, max_new_tokens=2000):
    """
    Analiza el corpus FLORES+ en gallego para identificar tokens óptimos
    """
    print("📥 Cargando corpus FLORES+ gallego...")
    
    # Cargar tokenizer original para análisis
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
    
    try:
        # Cargar todo el dataset FLORES+ gallego disponible
        ds_gl_dev = load_dataset("openlanguagedata/flores_plus", "glg_Latn", split="dev")
        ds_gl_devtest = load_dataset("openlanguagedata/flores_plus", "glg_Latn", split="devtest")
        
        # Combinar todos los textos gallegos
        all_galician_texts = []
        all_galician_texts.extend([item["text"] for item in ds_gl_dev])
        all_galician_texts.extend([item["text"] for item in ds_gl_devtest])
        
        print(f"✅ Corpus cargado: {len(all_galician_texts)} frases gallegas")
        
    except Exception as e:
        print(f"❌ Error cargando FLORES+: {e}")
        print("🔄 Usando corpus gallego de ejemplo...")
        all_galician_texts = [
            # Saludos y expresiones básicas
            "Ola, como estás?", "Que tal?", "Moitas grazas", "De nada",
            "Por favor", "Con permiso", "Disculpa", "Non hai problema",
            
            # Verbos comunes gallegos
            "Vou á casa", "Ven aquí", "Fala máis despacio", "Non entendo",
            "Podes axudarme?", "Onde está?", "Canto custa?", "Está ben",
            
            # Vida diaria
            "Hoxe fai bo tempo", "Onte choveu moito", "Mañá teño traballo",
            "A semana pasada fun de vacacións", "O ano que vem viaxarei",
            
            # Cultura galega
            "Galicia é fermosa", "Santiago de Compostela", "A Coruña",
            "Vigo é unha cidade grande", "O Camiño de Santiago",
            "A gaita galega", "As rías baixas", "O mar Cantábrico",
            
            # Comida gallega
            "O polbo á feira", "A empanada galega", "O caldo galego",
            "As vieiras", "O marisco", "A queimada", "O pan de centeo",
            
            # Expresiones gallegas únicas
            "Ai, miña nai!", "Que desastre!", "Está chovendo a cántaros",
            "Vou facer unha sesta", "Que aproveite!", "Saúde!",
        ] * 50  # Replicar para tener más datos
    
    return analyze_tokenization_patterns(tokenizer, all_galician_texts, min_freq, max_new_tokens)

def analyze_tokenization_patterns(tokenizer, texts: List[str], min_freq=3, max_new_tokens=2000):
    """
    Analiza patrones de tokenización problemáticos en textos gallegos
    """
    print("🔍 Analizando patrones de tokenización...")
    
    # Estadísticas
    word_stats = defaultdict(lambda: {
        'freq': 0, 
        'token_count': 0, 
        'has_unk': False,
        'efficiency_score': 0
    })
    
    # Palabras que aparecen frecuentemente pero se tokenizan mal
    problematic_words = Counter()
    
    # Patrones gallegos específicos a considerar
    galician_patterns = {
        # Contracciones comunes
        'contractions': ['do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas', 
                        'polo', 'pola', 'polos', 'polas', 'ao', 'á', 'aos', 'ás'],
        
        # Pronombres y artículos
        'pronouns': ['el', 'ela', 'eles', 'elas', 'me', 'te', 'se', 'nos', 'vos',
                    'meu', 'teu', 'seu', 'noso', 'voso', 'miña', 'túa', 'súa', 'nosa', 'vosa'],
        
        # Adverbios gallegos
        'adverbs': ['agora', 'despois', 'antes', 'sempre', 'nunca', 'onte', 'hoxe', 'mañá',
                   'aquí', 'alí', 'onde', 'como', 'cando', 'moito', 'pouco', 'ben', 'mal'],
        
        # Verbos gallegos comunes
        'verbs': ['son', 'está', 'estou', 'estás', 'están', 'teño', 'tes', 'ten', 'temos', 'tedes',
                 'vou', 'vas', 'vai', 'vamos', 'ides', 'van', 'fun', 'foches', 'foi', 'fomos'],
        
        # Sustantivos gallegos típicos
        'nouns': ['casa', 'tempo', 'día', 'noite', 'semana', 'mes', 'ano', 'xente', 'traballo',
                 'cidade', 'pobo', 'mar', 'río', 'monte', 'camiño', 'igrexa', 'praza'],
        
        # Adjetivos gallegos
        'adjectives': ['bo', 'boa', 'malo', 'mala', 'grande', 'pequeno', 'pequena', 
                      'fermoso', 'fermosa', 'novo', 'nova', 'vello', 'vella']
    }
    
    # Analizar cada texto
    for text in texts:
        # Limpiar y normalizar
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            if len(word) < 2:  # Ignorar palabras muy cortas
                continue
                
            # Tokenizar la palabra
            tokens = tokenizer.tokenize(word)
            has_unk = any('unk' in token.lower() for token in tokens)
            
            # Calcular score de eficiencia (menos tokens = mejor)
            efficiency_score = len(word) / max(len(tokens), 1)
            
            # Actualizar estadísticas
            word_stats[word]['freq'] += 1
            word_stats[word]['token_count'] = len(tokens)
            word_stats[word]['has_unk'] = has_unk
            word_stats[word]['efficiency_score'] = efficiency_score
            
            # Identificar palabras problemáticas
            if (len(tokens) > 2 or has_unk) and len(word) > 3:
                problematic_words[word] += 1
    
    print(f"📊 Analizadas {len(word_stats)} palabras únicas")
    print(f"⚠️  Identificadas {len(problematic_words)} palabras problemáticas")
    
    # Seleccionar nuevos tokens
    new_tokens = select_optimal_tokens(
        word_stats, 
        galician_patterns, 
        problematic_words, 
        min_freq, 
        max_new_tokens
    )
    
    return new_tokens, word_stats

def select_optimal_tokens(word_stats, galician_patterns, problematic_words, 
                         min_freq=3, max_new_tokens=2000):
    """
    Selecciona los tokens más óptimos para agregar al vocabulario
    """
    print("🎯 Seleccionando tokens óptimos...")
    
    candidates = []
    
    # 1. Palabras de patrones gallegos específicos (alta prioridad)
    pattern_words = set()
    for category, words in galician_patterns.items():
        pattern_words.update(words)
    
    for word in pattern_words:
        if word in word_stats and word_stats[word]['freq'] >= min_freq:
            score = word_stats[word]['freq'] * word_stats[word]['efficiency_score']
            candidates.append((word, score, 'pattern'))
    
    # 2. Palabras problemáticas frecuentes
    for word, freq in problematic_words.most_common():
        if freq >= min_freq and word not in pattern_words:
            if word in word_stats:
                score = freq * word_stats[word]['efficiency_score']
                candidates.append((word, score, 'problematic'))
    
    # 3. Palabras con baja eficiencia de tokenización pero frecuentes
    for word, stats in word_stats.items():
        if (stats['freq'] >= min_freq and 
            stats['efficiency_score'] < 0.7 and  # Baja eficiencia
            word not in pattern_words and 
            word not in problematic_words):
            score = stats['freq'] * (1 - stats['efficiency_score'])
            candidates.append((word, score, 'inefficient'))
    
    # Ordenar por score y seleccionar los mejores
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Balancear categorías
    selected_tokens = []
    category_counts = {'pattern': 0, 'problematic': 0, 'inefficient': 0}
    category_limits = {
        'pattern': max_new_tokens // 2,  # 50% para patrones gallegos
        'problematic': max_new_tokens // 3,  # 33% para problemáticas
        'inefficient': max_new_tokens // 6  # 17% para ineficientes
    }
    
    for word, score, category in candidates:
        if (len(selected_tokens) < max_new_tokens and 
            category_counts[category] < category_limits[category]):
            selected_tokens.append(word)
            category_counts[category] += 1
    
    print(f"✅ Seleccionados {len(selected_tokens)} tokens:")
    print(f"   📍 Patrones gallegos: {category_counts['pattern']}")
    print(f"   ⚠️  Problemáticos: {category_counts['problematic']}")
    print(f"   📉 Ineficientes: {category_counts['inefficient']}")
    
    # Mostrar ejemplos
    print(f"\n🔤 Ejemplos de tokens seleccionados:")
    for category in ['pattern', 'problematic', 'inefficient']:
        examples = [word for word, _, cat in candidates[:10] if cat == category]
        if examples:
            print(f"   {category}: {examples[:5]}")
    
    return selected_tokens

def expand_vocabulary_with_flores_analysis(model_name="Helsinki-NLP/opus-mt-tc-big-en-pt"):
    """
    Función principal para expandir vocabulario usando análisis de FLORES+
    """
    print("🚀 Iniciando expansión de vocabulario con análisis FLORES+...")
    
    # Cargar modelo y tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    original_vocab_size = len(tokenizer)
    
    print(f"📊 Vocabulario original: {original_vocab_size} tokens")
    
    # Analizar corpus gallego
    new_tokens, word_stats = analyze_galician_corpus_flores(
        min_freq=2,  # Más permisivo para capturar más variedad
        max_new_tokens=1500  # Limite razonable
    )
    
    if not new_tokens:
        print("❌ No se encontraron tokens para agregar")
        return tokenizer, model, []
    
    print(f"🆕 Agregando {len(new_tokens)} tokens al vocabulario...")
    
    # Expandir vocabulario
    tokenizer.add_tokens(new_tokens)
    
    # Redimensionar embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Inicialización inteligente de nuevos embeddings
    with torch.no_grad():
        # Obtener embeddings existentes
        existing_embeddings = model.get_input_embeddings().weight[:-len(new_tokens)]
        new_embeddings = model.get_input_embeddings().weight[-len(new_tokens):]
        
        # Calcular promedio por categorías similares si es posible
        avg_embedding = existing_embeddings.mean(dim=0)
        
        for i, token in enumerate(new_tokens):
            # Inicializar con promedio + ruido pequeño
            noise = torch.randn_like(avg_embedding) * 0.01
            new_embeddings[i] = avg_embedding + noise
    
    print(f"✅ Vocabulario expandido: {original_vocab_size} → {len(tokenizer)}")
    
    # Verificar mejoras
    print("\n🧪 Verificando mejoras en tokenización:")
    test_phrases = [
        "Ola, como estás hoxe?",
        "Gústame moito Galicia",
        "Vou á casa despois do traballo",
        "A empanada galega está moi boa"
    ]
    
    for phrase in test_phrases:
        tokens = tokenizer.tokenize(phrase)
        print(f"'{phrase}' → {len(tokens)} tokens: {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
    
    return tokenizer, model, new_tokens

# Función de prueba
if __name__ == "__main__":
    print("🔬 Probando expansión de vocabulario con FLORES+...")
    tokenizer, model, new_tokens = expand_vocabulary_with_flores_analysis()
    
    if new_tokens:
        print(f"\n📈 Resumen:")
        print(f"   • Tokens agregados: {len(new_tokens)}")
        print(f"   • Vocabulario final: {len(tokenizer)}")
        print(f"   • Ejemplos: {new_tokens[:10]}")
        
        # Guardar para uso posterior
        print("\n💾 Guardando tokenizer expandido...")
        tokenizer.save_pretrained("tokenizer_gallego_expandido")
        model.save_pretrained("modelo_gallego_expandido")
        print("✅ Guardado exitosamente!")
    else:
        print("❌ No se pudieron agregar tokens")