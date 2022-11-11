from cryptography.fernet import Fernet

key = Fernet.generate_key()

print(key)
key_file = open('key.key', 'wb')
key_file.write(key)
key_file.close()
