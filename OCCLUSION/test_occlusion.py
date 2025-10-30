"""
Test específico para verificar re-identificación tras oclusiones
"""
import sys
from pathlib import Path

print("=" * 80)
print("🔍 TEST: RE-IDENTIFICACIÓN TRAS OCLUSIONES")
print("=" * 80)
print()
print("📋 Configuración optimizada para mantener IDs tras oclusiones:")
print()
print("   🔹 MAX_LOST_FRAMES = 30 (mantiene tracks más tiempo)")
print("   🔹 MAHALANOBIS_THRESHOLD = 6.0 (más permisivo)")
print("   🔹 Distancia física adaptativa (crece con tiempo perdido)")
print("   🔹 Threshold adaptativo (1.5x para tracks perdidos >15 frames)")
print("   🔹 Q reducido (confía más en predicción)")
print("   🔹 Modo híbrido adaptativo (más Kalman para tracks establecidos)")
print()
print("-" * 80)
print("🎨 Observa los colores:")
print("   🟢 Verde = Vehículo detectado (con ID)")
print("   🔵 Azul claro = Predicción de Kalman (donde debería estar)")
print("   🔴 Naranja/Rojo = Vehículo en oclusión (predicción activa)")
print()
print("-" * 80)
print("✅ QUÉ ESPERAR:")
print("   • El bbox naranja (predicción) debe aparecer durante oclusión")
print("   • La predicción debe seguir la trayectoria esperada")
print("   • Cuando reaparece, debe recuperar el MISMO ID")
print()
print("❌ SI NO FUNCIONA:")
print("   • ID cambia = predicción está muy lejos de donde reaparece")
print("   • Activa debug: python main.py --debug")
print("   • Busca mensajes '[Match candidate]' y '[RE-ID]'")
print("   • Si no hay '[Match candidate]', distancia > threshold")
print()
print("-" * 80)
print("🐛 DEBUG MODE:")
print("   python main.py --debug")
print()
print("   Verás:")
print("   [Kalman] T<ID>: pos=(...) vel=(...) <- Estado predicho")
print("   [Match candidate] T<ID> <-> Det<N>: dist=X <- Candidatos")
print("   [✅ RE-ID] T<ID> (lost=N) <-> Det<M> <- Re-identificación exitosa")
print()
print("=" * 80)
print()

# Ejecutar
import main
try:
    main.main()
except KeyboardInterrupt:
    print("\n⚠️ Interrumpido por usuario")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
