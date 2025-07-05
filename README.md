# BWM THOR 2

Este repositório contém a implementação do método multicritério **BWM THOR 2**, desenvolvido para apoiar a seleção e ranqueamento de ações na bolsa brasileira (B3), visando a construção de carteiras de investimentos mais robustas.

## 📚 Descrição

O projeto integra o Best Worst Method (BWM) para definição dos pesos dos critérios de análise fundamentalista e o THOR 2 para classificação das alternativas considerando dados incertos e incompletos, conforme aplicado no estudo:

> "Aplicação do método BWM THOR 2 para seleção ótima de ações na B3 entre 2016 e 2019."

Os resultados mostram que a carteira construída a partir do ranking gerado pelo BWM THOR 2 superou o desempenho do índice Ibovespa e da poupança no período analisado.

## ⚙️ Como rodar

1. Clone o repositório:
   ```bash
   git clone https://github.com/Rhenan01/bwm-thor-2.git
   cd bwm-thor-2
   
2. (Opcional) Crie um ambiente virtual:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

3. Instale as dependências:
pip install -r requirements.txt

4. Execute o script principal (ajuste o nome se necessário):
python main.py

📖 Como citar
Se você usar este código em seu trabalho acadêmico, por favor cite como:
@misc{bwmthor2code,
  author = {Fabricio Tenorio, Rhenan Silva, Lucas Rocha},
  title = {Código BWM THOR 2 para análise multicritério de ações},
  howpublished = {\url{https://github.com/Rhenan01/bwm-thor-2}},
  year = {2025},
  note = {Acessado em: jul. 2025}
}


