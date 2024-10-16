# Mushroom Classification with Decision Trees and Random Forests

Questo progetto dimostra come utilizzare alberi decisionali e foreste casuali per classificare funghi utilizzando un dataset personalizzato. L'obiettivo è esplorare le performance di questi algoritmi di machine learning per compiti di classificazione categorica. Il progetto include il preprocessing dei dati, l'addestramento dei modelli, la sintonizzazione degli iperparametri e la valutazione dei modelli attraverso matrici di confusione.

## Dataset

Il dataset utilizzato in questo progetto è una versione modificata di un dataset per la classificazione dei funghi (`secondary_data.csv`). Il dataset contiene varie caratteristiche legate ai funghi e l'obiettivo è classificare se un determinato fungo è commestibile o velenoso.

## Struttura del Progetto

Le principali componenti del progetto includono:

- **Preprocessing dei Dati:** Gestione dei valori mancanti e preparazione del dataset per l'addestramento.
- **Implementazione del Modello:**
  - `DecisionTree`: Un classificatore ad albero decisionale personalizzato.
  - `RandomForest`: Un classificatore di foresta casuale personalizzato.
- **Funzioni di Supporto:**
  - `train_test_split_custom`: Per dividere il dataset in set di addestramento e test.
  - `hyperparameter_tuning`: Per la sintonizzazione degli iperparametri dei modelli.
  - `plot_confusion_matrix` e `compute_confusion_matrix`: Per visualizzare e calcolare le matrici di confusione.
  - `export_tree_to_graphviz`: Per esportare gli alberi decisionali per la visualizzazione.

## Requisiti

Le seguenti librerie Python sono utilizzate nel progetto:

```bash
pandas
numpy
matplotlib
graphviz
