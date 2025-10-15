import shared.utils as utils
from training import model as model

def startMenu():
    
    while(True):
    
        utils.printHead()
        
        print("| MODO DE OPERAÇÃO\n|\n| ( 1 ) - Realizar Inferência\n| ( 2 ) - Treinar Modelo\n| ( 0 ) - Finalizar")
        
        userInput = str(input("| Escolha: "))
        match userInput:
            case '1':
                return
            case '2':
                model.trainModel()
            case '0':
                print("\n| Saindo...")
                return
            case _:
                print("| Opção não definida!!!")

if __name__ == "__main__":
    startMenu()