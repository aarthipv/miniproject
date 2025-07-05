from difflib import SequenceMatcher

class OutputHandler:
    def display_results(self, original, reconstructed, highlighted, translation):
        return {
            'original': original,
            'reconstructed': reconstructed,
            'highlighted': highlighted,
            'translation': translation,
            'success': True
        }

    def render_to_user(self, data):
        # This could be extended for more complex rendering
        return data

    def highlight_insertions(self, original, reconstructed):
        try:
            matcher = SequenceMatcher(None, original.split(), reconstructed.split())
            highlighted = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    highlighted.extend(reconstructed.split()[j1:j2])
                elif tag == "insert" or tag == "replace":
                    highlighted.extend([
                        f'<mark class="reconstruction">{word}</mark>'
                        for word in reconstructed.split()[j1:j2]
                    ])
            return ' '.join(highlighted)
        except Exception:
            return reconstructed

    def format_for_display(self, text):
        # Demo: Add simple formatting (e.g., bold for demo)
        return f'<b>{text}</b>'

    def compute_metrics(self, original, reconstructed):
        # Demo: Compute word overlap and length difference
        orig_words = set(original.split())
        recon_words = set(reconstructed.split())
        overlap = len(orig_words & recon_words)
        total = len(orig_words) if orig_words else 1
        word_overlap = overlap / total
        length_diff = abs(len(original) - len(reconstructed))
        return {'word_overlap': word_overlap, 'length_difference': length_diff}

    def advanced_highlight(self, original, reconstructed):
        # Demo: Highlight differences at character level
        matcher = SequenceMatcher(None, original, reconstructed)
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                result.append(reconstructed[j1:j2])
            else:
                result.append(f'<mark class="reconstruction">{reconstructed[j1:j2]}</mark>')
        return ''.join(result) 