import  torch

class EarlyStopping:
    def __init__(self, patience, delta=0, mode='min',path = './Model/model.pth'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode 
        self.path = path

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_model(model,self.path)
        elif (self.mode == 'min' and current_score > self.best_score - self.delta) or \
             (self.mode == 'max' and current_score < self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_model(model,self.path)
            self.counter = 0

    def save_model(self, model,path = './model.pth'):
        torch.save(model.state_dict(), path)
