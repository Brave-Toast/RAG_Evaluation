import unittest

from backend.perturbations import KEYCAP_DIGITS, PerturbationSettings, perturb_query_with_emojis


class EmojiPerturbationTests(unittest.TestCase):
    def transform(self, text, **settings):
        return perturb_query_with_emojis(text, PerturbationSettings(**settings), 42)

    def test_words_mix_letters_emoji_and_numbers(self):
        query, record = self.transform('before great first today', emoji_style='substitute', emoji_count=4)
        self.assertEqual(query, 'b4️⃣ gr8️⃣ 1️⃣st 2️⃣day')
        self.assertEqual(record['input_query'], 'before great first today')
        self.assertEqual(record['output_query'], query)
        self.assertEqual(record['status'], 'applied')
        self.assertEqual(len(record['details']), 4)
        self.assertTrue(all(edit['kind'] == 'word_encoding' for edit in record['details']))

    def test_dates_numbers_separators_leading_zeros_and_punctuation_are_preserved(self):
        text = '10/09/2026, 2026-09-10; 09:05; 3.14!'
        query, record = self.transform(text, emoji_style='substitute', emoji_count=20)
        decoded = query
        for digit, keycap in KEYCAP_DIGITS.items():
            decoded = decoded.replace(keycap, digit)
        self.assertEqual(decoded, text)
        self.assertEqual(len(record['details']), 4)
        for edit in record['details']:
            self.assertEqual(edit['kind'], 'number_or_date')
            self.assertEqual(text[edit['offset']:edit['offset'] + len(edit['original'])], edit['original'])

    def test_letter_encoding_preserves_all_other_letters(self):
        query, record = self.transform('magazine', emoji_style='substitute', emoji_count=1)
        edit = record['details'][0]
        self.assertEqual(edit['kind'], 'letter_encoding')
        self.assertEqual(edit['original'], 'magazine')
        self.assertEqual(edit['replacement'], query)
        self.assertNotEqual(query, 'magazine')
        self.assertEqual(self.transform('magazine', emoji_style='substitute', emoji_count=1), (query, record))

    def test_insert_styles_keep_original_words_and_whitespace(self):
        text = 'Which  magazine\nstarted first?'
        for position in ['prefix', 'suffix', 'between_words']:
            with self.subTest(position=position):
                query, record = self.transform(text, emoji_style='insert', emoji_count=5, emoji_position=position)
                self.assertEqual(len(record['details']), 5)
                reconstructed = text
                additions = {}
                for edit in record['details']:
                    self.assertEqual(edit['kind'], 'insertion')
                    self.assertEqual(edit['position'], position)
                    if position == 'between_words':
                        self.assertTrue(text[edit['offset']].isspace())
                    additions[edit['offset']] = additions.get(edit['offset'], '') + edit['replacement']
                for offset in sorted(additions, reverse=True):
                    reconstructed = reconstructed[:offset] + additions[offset] + reconstructed[offset:]
                self.assertEqual(query, reconstructed)

    def test_existing_emoji_sequences_are_not_split(self):
        text = '🤔 1️⃣ 👨‍👩‍👧‍👦'
        encoded, record = self.transform(text, emoji_style='substitute', emoji_count=20)
        self.assertEqual(encoded, text)
        self.assertEqual(record['status'], 'skipped')
        inserted, _ = self.transform(text, emoji_style='insert', emoji_count=10, emoji_position='between_words')
        self.assertIn('1️⃣', inserted)
        self.assertIn('👨‍👩‍👧‍👦', inserted)

    def test_mixed_budget_repeatability_and_different_seeds(self):
        text = 'Which magazine started first before 2026?'
        query, record = self.transform(text, emoji_style='mixed', emoji_count=5)
        self.assertEqual(len(record['details']), 5)
        self.assertEqual(sum(edit['kind'] == 'insertion' for edit in record['details']), 3)
        self.assertEqual(self.transform(text, emoji_style='mixed', emoji_count=5), (query, record))
        results = {perturb_query_with_emojis(text, PerturbationSettings(emoji_count=5), seed)[0] for seed in range(5)}
        self.assertGreater(len(results), 1)

    def test_empty_or_ineligible_inputs_and_single_token_position(self):
        for style in ['insert', 'substitute', 'mixed']:
            query, record = self.transform(' \n ', emoji_style=style)
            self.assertEqual(query, ' \n ')
            self.assertEqual(record['status'], 'skipped')
        query, record = self.transform('sky GPT4 café', emoji_style='substitute')
        self.assertEqual(query, 'sky GPT4 café')
        self.assertEqual(record['status'], 'skipped')
        query, record = self.transform('sky', emoji_style='insert', emoji_position='between_words', emoji_count=1)
        self.assertTrue(query.startswith('sky '))
        self.assertEqual(record['details'][0]['position'], 'suffix')
        self.assertIn('No between-word boundary', record['summary'])


if __name__ == '__main__':
    unittest.main()
