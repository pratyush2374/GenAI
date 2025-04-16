import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print(f"Vocab size: {encoder.n_vocab}")  # Vocab size: 200019

text = "Hello bro this is Pratyush Sharma"
tokens = encoder.encode(text)
print(f"Tokens {tokens}")

received_tokens = [13225, 3714, 495, 382, 2284, 20204, 1776, 99835]
decrypted_tokens = encoder.decode(received_tokens)
print(f"Original tokens: {decrypted_tokens}")
