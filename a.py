def contar_a_y_d(texto):
    """Cuenta las letras 'a' y 'd' en un string."""
    cantidad_a = texto.lower().count('a')
    cantidad_d = texto.lower().count('d')

    return cantidad_a, cantidad_d


# Ejemplo de uso
texto = "dadaddaadadadaaaaaaaadaaadadadddadadddaaadadaaaadadaadadddddaaddddaadaaaddadadddddddadd"
a, d = contar_a_y_d(texto)
print(f"Letras 'a': {a}")
print(f"Letras 'd': {d}")