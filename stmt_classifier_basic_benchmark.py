INPUT_SENTENCES = [
    "<e>X</e> activates <e>Y</e>",
    "<e>X</e> inhibits <e>Y</e>",
    "<e>X</e> binds to <e>Y</e>",
    "<e>X</e> is a positive regulator of <e>Y</e>",
    "<e>X</e> is a negative regulator of <e>Y</e>",
    "<e>X</e> does not activate <e>Y</e>",
    "<e>X</e> does not inhibit <e>Y</e>",
    "<e>X</e> is not a positive regulator of <e>Y</e>",
    "<e>X</e> activates Y but it does not do anything to <e>Z</e>",
    "<e>X</e> inhibits Y but it does not do anything to <e>Z</e>",
    "<e>X</e> binds to Y but it does not do anything to <e>Z</e>",
    "<e>X</e> is a positive regulator of Y but it does not do anything to <e>Z</e>",
    "<e>X</e> is a negative regulator of Y but it does not do anything to <e>Z</e>"
]

from indra_bert.indra_stmt_classifier.model import IndraStmtClassifier
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    m = IndraStmtClassifier(model_path=args.model_path)
    for sentence in INPUT_SENTENCES:
        result = m.predict(sentence)
        print(f"Orginal text: {result['original_text']}\n"
              f"Decoded text (model's input decoded): {result['decoded_text']}\n"
              f"Predicted label: {result['predicted_label']}\n"
              f"Score: {result['confidence']}\n"
              f"Probabilities: {result['probabilities']}\n")

if __name__ == "__main__":
    main()
