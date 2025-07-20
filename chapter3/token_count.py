from ai21_tokenizer import Tokenizer

tokenizer = Tokenizer.get_tokenizer()
text = "Amazon Bedrock は AWS の生成AIサービスです。"

encoded_text = tokenizer.encode(text)
print(len(encoded_text))  # Print the number of tokens

encoded_text_en = tokenizer.encode("Amazon Bedrock is an AWS generative AI service.")
print(len(encoded_text_en))  # Print the number of tokens for English text
