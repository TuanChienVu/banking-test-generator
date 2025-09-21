#!/usr/bin/env python3
"""
Dataset Quality Analysis - Phân tích chi tiết chất lượng dataset
Author: Vũ Tuấn Chiến
Description: Đánh giá độ đa dạng, unique steps, duplication của dataset
"""

import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DatasetQualityAnalyzer:
    """Phân tích chất lượng dataset chi tiết"""
    
    def __init__(self):
        self.step_patterns = {
            'given': r'^Given\s+',
            'when': r'^When\s+',
            'then': r'^Then\s+',
            'and': r'^And\s+'
        }
        
    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load dataset từ file JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_all_steps(self, dataset: List[Dict]) -> List[str]:
        """Extract tất cả test steps từ dataset"""
        all_steps = []
        for item in dataset:
            steps = item.get('test_steps', [])
            if steps:
                all_steps.extend(steps)
            elif 'target_text' in item:
                # Parse từ target_text nếu không có test_steps
                steps = item['target_text'].split('\n')
                all_steps.extend(steps)
        return all_steps
    
    def analyze_step_diversity(self, steps: List[str]) -> Dict:
        """Phân tích độ đa dạng của steps"""
        # Basic stats
        total_steps = len(steps)
        unique_steps = set(steps)
        unique_count = len(unique_steps)
        
        # Step type distribution
        step_types = Counter()
        for step in steps:
            for step_type, pattern in self.step_patterns.items():
                if re.match(pattern, step, re.IGNORECASE):
                    step_types[step_type] += 1
                    break
            else:
                step_types['other'] += 1
        
        # Word-level diversity
        all_words = []
        for step in steps:
            words = step.lower().split()
            all_words.extend(words)
        
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        # Most common steps
        step_counter = Counter(steps)
        most_common_steps = step_counter.most_common(10)
        
        # Calculate diversity score
        diversity_score = unique_count / total_steps if total_steps > 0 else 0
        
        return {
            'total_steps': total_steps,
            'unique_steps': unique_count,
            'diversity_score': diversity_score,
            'step_types': dict(step_types),
            'total_words': total_words,
            'unique_words': unique_words,
            'word_diversity': unique_words / total_words if total_words > 0 else 0,
            'most_common_steps': most_common_steps,
            'duplication_rate': 1 - diversity_score
        }
    
    def analyze_test_case_similarity(self, dataset: List[Dict], sample_size: int = 100) -> Dict:
        """Phân tích độ tương đồng giữa các test cases"""
        # Extract test cases as text
        test_cases = []
        for item in dataset[:sample_size]:  # Sample for performance
            if 'test_steps' in item:
                test_case = '\n'.join(item['test_steps'])
            elif 'target_text' in item:
                test_case = item['target_text']
            else:
                continue
            test_cases.append(test_case)
        
        if len(test_cases) < 2:
            return {'error': 'Not enough test cases for similarity analysis'}
        
        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(test_cases)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Get similarity statistics
        # Exclude diagonal (self-similarity)
        n = len(similarity_matrix)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(similarity_matrix[i][j])
        
        if not similarities:
            return {'error': 'Could not calculate similarities'}
        
        return {
            'num_test_cases_analyzed': len(test_cases),
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'std_similarity': np.std(similarities),
            'highly_similar_pairs': sum(1 for s in similarities if s > 0.8) / len(similarities),
            'very_different_pairs': sum(1 for s in similarities if s < 0.2) / len(similarities)
        }
    
    def detect_exact_duplicates(self, dataset: List[Dict]) -> Dict:
        """Detect exact duplicates trong dataset"""
        test_case_hashes = {}
        duplicates = []
        
        for idx, item in enumerate(dataset):
            # Create test case representation
            if 'test_steps' in item:
                test_case = '\n'.join(item['test_steps'])
            elif 'target_text' in item:
                test_case = item['target_text']
            else:
                continue
            
            # Calculate hash
            test_hash = hashlib.md5(test_case.encode()).hexdigest()
            
            if test_hash in test_case_hashes:
                duplicates.append({
                    'original_idx': test_case_hashes[test_hash],
                    'duplicate_idx': idx,
                    'test_id': item.get('test_id', 'N/A')
                })
            else:
                test_case_hashes[test_hash] = idx
        
        return {
            'total_test_cases': len(dataset),
            'unique_test_cases': len(test_case_hashes),
            'exact_duplicates': len(duplicates),
            'duplication_rate': len(duplicates) / len(dataset) if dataset else 0,
            'duplicate_examples': duplicates[:5]  # Show first 5 examples
        }
    
    def analyze_feature_coverage(self, dataset: List[Dict]) -> Dict:
        """Phân tích coverage của các features"""
        feature_counter = Counter()
        scenario_counter = Counter()
        priority_counter = Counter()
        
        for item in dataset:
            feature_counter[item.get('feature', 'unknown')] += 1
            scenario_counter[item.get('scenario_type', 'unknown')] += 1
            priority_counter[item.get('priority', 'unknown')] += 1
        
        return {
            'features': dict(feature_counter),
            'num_features': len(feature_counter),
            'scenarios': dict(scenario_counter),
            'priorities': dict(priority_counter),
            'feature_balance': np.std(list(feature_counter.values())) / np.mean(list(feature_counter.values())) if feature_counter else 0
        }
    
    def analyze_step_patterns(self, steps: List[str]) -> Dict:
        """Phân tích patterns trong steps"""
        # Common action verbs
        action_verbs = ['click', 'tap', 'enter', 'input', 'select', 'verify', 'check', 'navigate', 'login', 'logout']
        verb_usage = Counter()
        
        # Common UI elements
        ui_elements = ['button', 'field', 'page', 'screen', 'link', 'menu', 'form', 'dropdown', 'checkbox', 'radio']
        element_usage = Counter()
        
        for step in steps:
            step_lower = step.lower()
            
            # Count verb usage
            for verb in action_verbs:
                if verb in step_lower:
                    verb_usage[verb] += 1
            
            # Count UI element usage
            for element in ui_elements:
                if element in step_lower:
                    element_usage[element] += 1
        
        # Pattern variety
        unique_patterns = set()
        for step in steps:
            # Extract pattern (replace numbers, amounts with placeholders)
            pattern = re.sub(r'\$[\d,]+\.?\d*', '$AMOUNT', step)
            pattern = re.sub(r'\b\d+\b', 'NUM', pattern)
            pattern = re.sub(r'"[^"]*"', '"TEXT"', pattern)
            unique_patterns.add(pattern)
        
        return {
            'common_verbs': dict(verb_usage.most_common(10)),
            'common_ui_elements': dict(element_usage.most_common(10)),
            'unique_patterns': len(unique_patterns),
            'pattern_diversity': len(unique_patterns) / len(steps) if steps else 0
        }
    
    def generate_comprehensive_report(self, dataset_path: str) -> Dict:
        """Generate comprehensive quality report"""
        print(f"\n📊 Analyzing dataset: {Path(dataset_path).name}")
        print("="*60)
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Extract all steps
        all_steps = self.extract_all_steps(dataset)
        
        # Run all analyses
        report = {
            'file': Path(dataset_path).name,
            'basic_stats': {
                'total_items': len(dataset),
                'total_steps': len(all_steps),
                'avg_steps_per_item': len(all_steps) / len(dataset) if dataset else 0
            },
            'step_diversity': self.analyze_step_diversity(all_steps),
            'exact_duplicates': self.detect_exact_duplicates(dataset),
            'similarity_analysis': self.analyze_test_case_similarity(dataset),
            'feature_coverage': self.analyze_feature_coverage(dataset),
            'step_patterns': self.analyze_step_patterns(all_steps)
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted report"""
        print(f"\n📁 File: {report['file']}")
        print("-"*60)
        
        # Basic stats
        print("\n📈 BASIC STATISTICS:")
        stats = report['basic_stats']
        print(f"  • Total items: {stats['total_items']:,}")
        print(f"  • Total steps: {stats['total_steps']:,}")
        print(f"  • Avg steps/item: {stats['avg_steps_per_item']:.1f}")
        
        # Step diversity
        print("\n🎯 STEP DIVERSITY:")
        diversity = report['step_diversity']
        print(f"  • Unique steps: {diversity['unique_steps']:,} / {diversity['total_steps']:,}")
        print(f"  • Diversity score: {diversity['diversity_score']:.3f}")
        print(f"  • Duplication rate: {diversity['duplication_rate']:.1%}")
        print(f"  • Word diversity: {diversity['word_diversity']:.3f}")
        print(f"  • Step types: {diversity['step_types']}")
        
        # Duplicates
        print("\n🔍 EXACT DUPLICATES:")
        duplicates = report['exact_duplicates']
        print(f"  • Unique test cases: {duplicates['unique_test_cases']:,} / {duplicates['total_test_cases']:,}")
        print(f"  • Exact duplicates: {duplicates['exact_duplicates']:,}")
        print(f"  • Duplication rate: {duplicates['duplication_rate']:.1%}")
        
        # Similarity
        print("\n📊 SIMILARITY ANALYSIS (sample):")
        similarity = report['similarity_analysis']
        if 'error' not in similarity:
            print(f"  • Avg similarity: {similarity['avg_similarity']:.3f}")
            print(f"  • Highly similar pairs: {similarity['highly_similar_pairs']:.1%}")
            print(f"  • Very different pairs: {similarity['very_different_pairs']:.1%}")
        else:
            print(f"  • Error: {similarity['error']}")
        
        # Feature coverage
        print("\n🎨 FEATURE COVERAGE:")
        features = report['feature_coverage']
        print(f"  • Number of features: {features['num_features']}")
        print(f"  • Feature balance (CV): {features['feature_balance']:.3f}")
        print(f"  • Top 3 features: {dict(Counter(features['features']).most_common(3))}")
        
        # Step patterns
        print("\n🔧 STEP PATTERNS:")
        patterns = report['step_patterns']
        print(f"  • Unique patterns: {patterns['unique_patterns']:,}")
        print(f"  • Pattern diversity: {patterns['pattern_diversity']:.3f}")
        print(f"  • Top verbs: {list(patterns['common_verbs'].keys())[:5]}")
        print(f"  • Top UI elements: {list(patterns['common_ui_elements'].keys())[:5]}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dataset quality analysis")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).parent.parent / "datasets" / "final" / "testcase_generation" / "diverse_v4_natural_diversity_v1"),
        help="Path to dataset directory (containing JSON files). Defaults to latest V4 path.",
    )
    args = parser.parse_args()

    analyzer = DatasetQualityAnalyzer()

    # Analyze dataset files from provided directory
    dataset_dir = Path(args.dataset_dir)

    # Preferred files (V4 layout)
    preferred = [dataset_dir / "train.json", dataset_dir / "val.json", dataset_dir / "test.json"]
    files_to_analyze = [p for p in preferred if p.exists()]

    if not files_to_analyze:
        # Fallback: pattern match train*/val*/test* in the directory
        all_json_files = sorted(dataset_dir.glob("*.json"))
        for json_file in all_json_files:
            filename = json_file.name.lower()
            if any(x in filename for x in ["metadata", "comparison", "temp"]):
                continue
            if filename.startswith("train") or filename.startswith("val") or filename.startswith("test"):
                files_to_analyze.append(json_file)

    if not files_to_analyze:
        # Last resort: analyze all JSON except metadata/comparison/temp
        files_to_analyze = [
            f for f in sorted(dataset_dir.glob("*.json"))
            if all(x not in f.name.lower() for x in ["metadata", "comparison", "temp"])
        ]

    all_reports = []

    print("=" * 60)
    print("🔬 COMPREHENSIVE DATASET QUALITY ANALYSIS")
    print("=" * 60)

    for filepath in files_to_analyze:
        if filepath.exists():
            report = analyzer.generate_comprehensive_report(str(filepath))
            analyzer.print_report(report)
            all_reports.append(report)
        else:
            print(f"\n⚠️ File not found: {filepath.name}")

    # Summary comparison
    if all_reports:
        print("\n" + "=" * 60)
        print("📊 DATASET COMPARISON SUMMARY")
        print("=" * 60)

        print("\n{:<30} {:>10} {:>10} {:>12} {:>12}".format(
            "Dataset", "Items", "Unique", "Diversity", "Dup Rate"
        ))
        print("-" * 80)

        for report in all_reports:
            filename = report['file'][:28]
            items = report['basic_stats']['total_items']
            unique = report['exact_duplicates']['unique_test_cases']
            diversity = report['step_diversity']['diversity_score']
            dup_rate = report['exact_duplicates']['duplication_rate']

            print("{:<30} {:>10,} {:>10,} {:>12.3f} {:>12.1%}".format(
                filename, items, unique, diversity, dup_rate
            ))

        # Find best and worst
        best_diversity = max(all_reports, key=lambda x: x['step_diversity']['diversity_score'])
        worst_diversity = min(all_reports, key=lambda x: x['step_diversity']['diversity_score'])

        print("\n📈 INSIGHTS:")
        print(f"  • Best diversity: {best_diversity['file']} ({best_diversity['step_diversity']['diversity_score']:.3f})")
        print(f"  • Worst diversity: {worst_diversity['file']} ({worst_diversity['step_diversity']['diversity_score']:.3f})")

        # Calculate overall stats
        total_items = sum(r['basic_stats']['total_items'] for r in all_reports)
        total_unique = sum(r['exact_duplicates']['unique_test_cases'] for r in all_reports)

        print(f"\n  • Total items across all: {total_items:,}")
        print(f"  • Total unique across all: {total_unique:,}")
        print(f"  • Overall duplication: {(1 - total_unique/total_items)*100:.1f}%")

if __name__ == "__main__":
    # Install required packages if not available
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess
        subprocess.call(['pip', 'install', 'scikit-learn'])
        from sklearn.feature_extraction.text import TfidfVectorizer
    
    main()
