[![-----------------------------------------------------](
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/alpardayalman?tab=repositories)

## DSLR

[![-----------------------------------------------------](
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/alpardayalman?tab=repositories)

## Developers
<table>
  <tbody>
    <tr align="center" >
      <td align="center" valign="top" width="20%"><a href="https://github.com/timurci"><img src="https://avatars.githubusercontent.com/u/83092851?v=4" width="100px;" alt="Timur Çakmakoğlu"/><br /><sub><b>Timur Çakmakoğlu</b></sub></a><br /> 
      </td>
      <td align="center" valign="top" width="30%"><a href="https://github.com/alpardayalman"><img src="https://avatars.githubusercontent.com/u/82611850?v=4" width="100px;" alt="Alp A. Yalman"/><br /><sub><b>Alp A. Yalman</b></sub></a><br />
      </td>
    </tr>
  </table>
</tbody>

[![-----------------------------------------------------](
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/alpardayalman?tab=repositories)

## DSLR (again)

In this project DataScience x Logistic Regression, you will continue your exploration of
Machine Learning by discovering different tools.

The use of the term DataScience in the title will be clearly considered by some to be
abusive. That is true. We do not pretend to give you all the basics of DataScience in this
topic. The subject is vast. We will only see here some bases which seemed to us useful
for data exploration before sending it to the machine learning algorithm.

You will implement a **linear classification** model, as a continuation of the subject lin-
ear regression : a logistic regression. We also encourage you a lot to create a machine
learning toolkit while you will move along the branch.

### Summarizing:

•You will learn how to read a data set, to visualize it in different ways, to select and
clean unnecessary information from your data.

•You will train a logistic regression that will solve classification problem.

[![-----------------------------------------------------](
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/alpardayalman?tab=repositories)

```bash
  git clone git@github.com:alpardayalman/dslr.git
```

Virtual env

```bash
  python3 -m venv venv
  source venv/bin/activate
```

Install modules

```bash
  pip install -r requirements.txt
```

To Train

```bash
  python logreg_train.py assets/datasets/dataset_train.csv -l "Hogwarts House"
```

To Test 

```bash
  logreg_predict.py assets/datasets/dataset_test.csv weights.csv -l "Hogwarts House" -o {name_of_output_file}
```
