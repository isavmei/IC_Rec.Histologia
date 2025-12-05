# IC_Rec.Histologia
Estudo para o trabalho de pesquisa do IC, para processamento da histologia de células sanguíneas.

🧩 Etapa 2 – Coleta e Organização do Dataset

Acessar a plataforma Kaggle e baixar o BCCD Dataset (Blood Cell Count and Detection).

Verificar a integridade e rotulação das imagens (neutrófilos, linfócitos, monócitos e eosinófilos).

Excluir imagens duplicadas, com baixa resolução ou rotulagem incorreta.

Garantir que os dados estão anonimizados e sem informações sensíveis.

Criar pastas separadas por classe celular (ex.: /data/train/lymphocyte/, /data/test/monocyte/).








⚙️ Etapa 3 – Pré-processamento das Imagens

Padronizar a resolução e o formato das imagens (ex.: 256×256 px, .jpg).

Aplicar:

Normalização de intensidade;

Conversão de canais de cor (RGB → escala adequada);

Remoção de ruído (filtros OpenCV);

Aumento de dados (data augmentation).

Extrair características com OpenCV e NumPy:

Cor: média e variância dos canais RGB;

Textura: LBP, GLCM;

Forma: área, perímetro, circularidade;

Distribuição: histograma de intensidade e densidade de pixels.

Salvar as features em arquivos .csv ou .npy para uso nos modelos.








dataset_processed/         

│

├─ CORROMPIDAS/            

├─ BAIXA_RES/              

├─ DUPLICADAS/             

├─ RAW_CLEANED/           

│   ├─ train/

│   │   ├─ class1/

│   │   └─ class2/

│   └─ test/

│       ├─ class1/

│       └─ class2/

└─ features/              

    ├─ features.csv        
    
    ├─ features.npy        
    
    └─ labels.npy          
    
