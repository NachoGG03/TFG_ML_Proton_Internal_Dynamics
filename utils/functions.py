### SOME FUNCTIONS USED IN THE PROJECT ###

## Function to translate the pid of the partons to their corresponding LaTeX symbols.
def pid_to_latex(pid):
    translate = {21: "g", 1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t"}
    flav = translate[abs(pid)]
    if pid < 0:
        flav = rf"$\bar{{{flav}}}$"
    return flav
