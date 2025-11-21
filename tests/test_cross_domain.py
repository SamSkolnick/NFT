
import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ResearchEval import ResearchEvaluator

class TestCrossDomainEval(unittest.TestCase):
    def setUp(self):
        self.evaluator = ResearchEvaluator(anthropic_api_key="dummy")

    def test_combine_scores_high_cross_domain(self):
        # Mock scores
        process_score = {'overall': 0.8}
        impact_score = {
            'overall': 0.8,
            'breakdown': {
                'cross_domain': {
                    'bonus': 0.2,
                    'analogy_score': 0.9
                }
            }
        }
        
        # Expected: 
        # Base = 0.2*0.8 + 0.8*0.8 = 0.16 + 0.64 = 0.80
        # Bonus (both good) = +0.1 -> 0.90
        # Analogy Bonus (>0.6) = +0.15 -> 1.05 -> 1.0
        # Cross Domain Bonus = +0.2 -> 1.2 -> 1.0
        
        final_score = self.evaluator._combine_scores(process_score, impact_score)
        self.assertEqual(final_score, 1.0)

    def test_combine_scores_low_cross_domain(self):
        process_score = {'overall': 0.8}
        impact_score = {
            'overall': 0.5,
            'breakdown': {
                'cross_domain': {
                    'bonus': 0.0,
                    'analogy_score': 0.2
                }
            }
        }
        
        # Base = 0.2*0.8 + 0.8*0.5 = 0.16 + 0.40 = 0.56
        # No bonuses
        
        final_score = self.evaluator._combine_scores(process_score, impact_score)
        self.assertAlmostEqual(final_score, 0.56)

    @patch("ResearchEval.ResearchEvaluator._read_all_research_files")
    @patch("ResearchEval.ResearchEvaluator._extract_citations")
    def test_evaluate_cross_domain_transfer_parsing(self, mock_citations, mock_read):
        mock_read.return_value = "Research content..."
        mock_citations.return_value = [{"title": "Paper A", "year": 2023}]
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "same_domain_papers": [],
            "cross_domain_papers": [],
            "analogy_quality": {
                "score": 0.85,
                "reasoning": "Great analogy",
                "mapping_examples": ["A -> B"]
            },
            "cross_domain_usage": {
                "used": True,
                "how": "Used it"
            },
            "score": 0.9
        }))]
        
        self.evaluator.client.messages.create = MagicMock(return_value=mock_response)
        
        result = self.evaluator._evaluate_cross_domain_transfer("path", "domain")
        
        self.assertEqual(result['analogy_score'], 0.85)
        # Bonus: used(0.1) + analogy>0.7(0.1) = 0.2
        self.assertEqual(result['bonus'], 0.2)

if __name__ == "__main__":
    unittest.main()
