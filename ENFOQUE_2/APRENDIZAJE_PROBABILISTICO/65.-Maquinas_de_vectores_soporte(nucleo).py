"""
Máquinas de Vectores Soporte (SVM) con Kernel
Aprendizaje Probabilístico
Algoritmo de clasificación lineal y no-lineal
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ===== PARTE 1: SVM LINEAL =====
def svm_lineal():
    """
    Máquina de Vectores Soporte Lineal
    Busca el hiperplano que maximiza el margen entre clases
    """
    print("="*60)
    print("SVM LINEAL")
    print("="*60)
    
    # Cargar dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # Usar solo 2 características para visualización
    y = iris.target
    
    # Filtrar solo 2 clases
    mask = y != 2
    X = X[mask]
    y = y[mask]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entrenar SVM lineal
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    
    # Predicciones
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precisión: {accuracy:.4f}")
    print(f"Número de vectores soporte: {len(svm.support_vectors_)}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    return svm, scaler, X_train, X_test, y_train, y_test

# ===== PARTE 2: SVM CON KERNEL RBF =====
def svm_kernel_rbf():
    """
    Máquina de Vectores Soporte con Kernel RBF
    Para datos no linealmente separables
    """
    print("\n" + "="*60)
    print("SVM CON KERNEL RBF (Radial Basis Function)")
    print("="*60)
    
    # Cargar dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    
    # Filtrar solo 2 clases
    mask = y != 2
    X = X[mask]
    y = y[mask]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entrenar SVM con kernel RBF
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X_train, y_train)
    
    # Predicciones
    y_pred = svm_rbf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precisión: {accuracy:.4f}")
    print(f"Número de vectores soporte: {len(svm_rbf.support_vectors_)}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    return svm_rbf, scaler, X_train, X_test, y_train, y_test

# ===== PARTE 3: SVM CON KERNEL POLINÓMICO =====
def svm_kernel_polinomico():
    """
    Máquina de Vectores Soporte con Kernel Polinómico
    """
    print("\n" + "="*60)
    print("SVM CON KERNEL POLINÓMICO")
    print("="*60)
    
    # Cargar dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    
    # Filtrar solo 2 clases
    mask = y != 2
    X = X[mask]
    y = y[mask]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Entrenar SVM con kernel polinómico
    svm_poly = SVC(kernel='poly', degree=3, C=1.0)
    svm_poly.fit(X_train, y_train)
    
    # Predicciones
    y_pred = svm_poly.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precisión: {accuracy:.4f}")
    print(f"Número de vectores soporte: {len(svm_poly.support_vectors_)}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    return svm_poly, scaler, X_train, X_test, y_train, y_test

# ===== PARTE 4: COMPARACIÓN DE KERNELS =====
def comparar_kernels():
    """
    Compara el desempeño de diferentes kernels
    """
    print("\n" + "="*60)
    print("COMPARACIÓN DE KERNELS")
    print("="*60)
    
    # Cargar dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Usar todas las características
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    resultados = {}
    
    for kernel in kernels:
        svm = SVC(kernel=kernel, C=1.0)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        resultados[kernel] = {
            'accuracy': accuracy,
            'support_vectors': len(svm.support_vectors_)
        }
        
        print(f"\nKernel: {kernel.upper()}")
        print(f"  Precisión: {accuracy:.4f}")
        print(f"  Vectores soporte: {resultados[kernel]['support_vectors']}")
    
    return resultados

# ===== PARTE 5: OPTIMIZACIÓN DE PARÁMETROS =====
def optimizar_parametros_C():
    """
    Optimiza el parámetro de regularización C
    """
    print("\n" + "="*60)
    print("OPTIMIZACIÓN DEL PARÁMETRO C")
    print("="*60)
    
    # Cargar dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Prueba con diferentes valores de C
    valores_C = [0.1, 1, 10, 100]
    resultados = {}
    
    print("Kernel: RBF")
    for c in valores_C:
        svm = SVC(kernel='rbf', C=c)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        resultados[c] = accuracy
        print(f"  C={c:>4}: Precisión = {accuracy:.4f}")
    
    return resultados

# ===== MAIN =====
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# MÁQUINAS DE VECTORES SOPORTE (SVM)")
    print("# Aprendizaje Probabilístico")
    print("#"*60)
    
    # Ejecutar algoritmos
    svm_lineal()
    svm_kernel_rbf()
    svm_kernel_polinomico()
    comparar_kernels()
    optimizar_parametros_C()
    
    print("\n" + "#"*60)
    print("# ANÁLISIS COMPLETADO")
    print("#"*60)
