import matplotlib.pyplot as plt
import numpy as np

# Inizializzazione delle liste per i dati
block_sizes = []
monolithic_times = []
grid_stride_times = []

# Lettura e parsing dei dati dal file results.txt
with open('results.txt', 'r') as f:
    for line in f:
        if line.startswith('Monolithic') or line.startswith('GridStride'):
            parts = line.strip().split(',')
            version, block_size, time = parts
            if version == 'Monolithic':
                monolithic_times.append(float(time))
            else:
                grid_stride_times.append(float(time))
            if block_size not in block_sizes:
                block_sizes.append(block_size)

# Preparazione dei dati per il grafico
x = np.arange(len(block_sizes))  # Posizioni per le etichette delle dimensioni del blocco
width = 0.35  # Larghezza delle barre

# Creazione del grafico
fig, ax = plt.subplots(figsize=(12, 6))

# Barre per la versione Monolitica
rects1 = ax.bar(x - width/2, monolithic_times, width, label='Monolithic')

# Barre per la versione Grid-Stride
rects2 = ax.bar(x + width/2, grid_stride_times, width, label='Grid-Stride')

# Impostazione delle etichette e titolo del grafico
ax.set_ylabel('Execution Time (ms)')
ax.set_xlabel('Block Sizes')
ax.set_title('Comparison of Execution Times for Different Block Sizes')
ax.set_xticks(x)
ax.set_xticklabels(block_sizes)
ax.legend()

# Rotazione delle etichette sull'asse X per una migliore leggibilità
plt.xticks(rotation=45)

# Ottimizzazione della visualizzazione del grafico
plt.tight_layout()

# Funzione per aggiungere le etichette sopra le barre
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',  # Formattazione del testo
                    xy=(rect.get_x() + rect.get_width() / 2, height),  # Posizione del testo
                    xytext=(0, 3),  # Offset del testo
                    textcoords="offset points",  # Coordinate relative al punto
                    ha='center', va='bottom')

# Aggiunta delle etichette sopra le barre
autolabel(rects1)
autolabel(rects2)

# Salvataggio del grafico in formato PDF
plt.savefig('matrix_multiplication_comparison.pdf')

# Visualizzazione del grafico
plt.show()
