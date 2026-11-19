"""Export within-library comparisons for configurations with all four measured folds."""
import pandas as pd


def paired_results(df, folds, out):
    wide=df.pivot(index=['lib','design','stage'],columns='fold',values='mrrmse').reindex(columns=folds).dropna()
    rows=[]
    def compare(lib, question, left, right):
        a=(lib, *left); b=(lib, *right)
        if a not in wide.index or b not in wide.index:
            return
        delta=wide.loc[b]-wide.loc[a]
        rows.append(dict(library=lib,comparison=question,reference='/'.join(left),candidate='/'.join(right),**delta.to_dict(),mean_change=delta.mean(),improved_folds=int((delta<0).sum())))
    for lib in ['xgb','pyboost']:
        for d,e in [('A','C'),('B','D')]:compare(lib,'Add baseline RNA',(d,'base'),(e,'base'))
        for d,e in [('A','B'),('C','D')]:compare(lib,'Apply output PCA160',(d,'base'),(e,'base'))
        for d in 'ABCD':
            compare(lib,'Structure at fixed schedule',(d,'base'),(d,'structure'))
            compare(lib,'Learning rate and rounds at fixed structure',(d,'structure'),(d,'schedule'))
    compare('xgb','Tree strategy at T09 settings',('B','T09'),('B','strategy'))
    table=pd.DataFrame(rows);table.to_csv(out/'paired_differences.csv',index=False)
    return ['<h2>Matched comparisons</h2><p>Each value is the candidate score minus the reference on the same validation fold. Negative values favor the candidate. XGBoost structure changes depth and L2 together. PyBoost structure changes depth alone. Schedule changes learning rate and rounds together.</p>',table.to_html(index=False,float_format=lambda x:f'{x:.6f}'),'<p>At baseline with PCA160, adding RNA improved PyBoost in all four folds. For XGBoost, CD4 and CD8 improved while NK and Treg worsened. The XGBoost shared-tree control improved CD8 and Treg slightly but lost more on NK and CD4. The two libraries also differ in training implementation and selected parameters, so their final score gap cannot isolate the effect of tree strategy.</p>']
