"""Analyze continuous-time SISO transfer functions from a Slovenian CLI.

The program accepts numerator and denominator coefficients in descending
powers of ``s`` and provides five common control-system visualizations:
Bode, step response, impulse response, pole-zero map, and root locus.
"""

import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def menu():
    """Display the analysis menu and return the selected option."""
    return input("""
Izberi:

1 Bode Diagram
2 Step Response
3 Impulse Response
4 Pole-Zero Map
5 Root Locus
6 Exit

Izbira: """).strip()


def bode_plot_error(num, den):
    """Validate transfer-function coefficient lists.

    The leading coefficient of each polynomial must be nonzero so that the
    list length represents the polynomial order unambiguously. NaN and
    infinite values are rejected because numerical analysis routines cannot
    produce meaningful results from them.

    Parameters
    ----------
    num : list of float
        Numerator coefficients in descending powers of ``s``.
    den : list of float
        Denominator coefficients in descending powers of ``s``.

    Raises
    ------
    ValueError
        If either list is empty or contains an invalid leading/value entry.
    """

    if not num:
        raise ValueError("Numerator cannot be empty.")

    if not all(np.isfinite(x) for x in num + den):
        raise ValueError("Coefficients must be finite numbers.")

    if num[0] == 0:
        raise ValueError("The first numerator coefficient cannot be zero.")

    if not den:
        raise ValueError("Denominator cannot be empty.")

    if den[0] == 0:
        raise ValueError("The first denominator coefficient cannot be zero.")


def vnos_koeficientov():
    """Read, validate, and display a transfer function entered by the user.

    Invalid input returns the user to the coefficient prompts. The function
    returns only after both polynomials pass validation.
    """
    print("=" * 40)
    print(" CONTROL ANALYSIS TOOL")
    print("=" * 40)
    while True:
        try:

            print("Enter the numerator coefficients (separated by a space):")
            num = list(map(float, input("Numerator: ").split()))

            print("\nEnter the denominator coefficients (separated by a space):")
            den = list(map(float, input("Denominator: ").split()))
            print('\n')

            bode_plot_error(num, den)

            # Continue only after both coefficient lists are valid.
            break

        except ValueError as error:
            print(f"\nError: {error}")
            print("Please enter the coefficients again.\n")

    num_str = polynomial_to_string(num)
    den_str = polynomial_to_string(den)

    # Match the fraction bar to the longer polynomial for aligned output.
    line_length = max(len(num_str), len(den_str))
    line = "-" * line_length

    prefix = "G(s) = "

    # Print the transfer function as a centered, three-line fraction.
    print(" " * len(prefix) + num_str.center(line_length))
    print(prefix + line)
    print(" " * len(prefix) + den_str.center(line_length))

    return num, den


def polynomial_to_string(coeffs):
    """Convert polynomial coefficients to a readable expression in ``s``.

    For example, ``[1, 0, 0.4]`` becomes ``"s^2 + 0.4"``. Zero-valued terms
    are omitted and coefficients equal to one are hidden before ``s``.

    Parameters
    ----------
    coeffs : sequence of float
        Polynomial coefficients in descending powers of ``s``.

    Returns
    -------
    str
        Polynomial formatted for terminal output.
    """
    degree = len(coeffs) - 1
    terms = []

    for i, coeff in enumerate(coeffs):

        # The coefficient position determines the corresponding power of s.
        power = degree - i
        power_text = "^" + str(power)

        # Omit zero-valued terms from the displayed polynomial.
        if coeff == 0:
            continue

        # Avoid displaying integer-valued coefficients with a trailing .0.
        coeff = int(coeff) if float(coeff).is_integer() else coeff

        if power > 1:
            if coeff == 1:
                terms.append(f"s{power_text}")
            elif coeff == -1:
                terms.append(f"-s{power_text}")
            else:
                terms.append(f"{coeff}s{power_text}")

        elif power == 1:
            if coeff == 1:
                terms.append("s")
            elif coeff == -1:
                terms.append("-s")
            else:
                terms.append(f"{coeff}s")

        else:
            terms.append(f"{coeff}")
    polynomial = " + ".join(terms)

    # Replace awkward additions of negative terms with subtraction.
    polynomial = polynomial.replace("+ -", "- ")

    return polynomial


def bode_diagram(num, den):
    """Plot Bode magnitude and phase over 0.01 to 1000 rad/s."""
    # Logarithmic sampling gives even visual detail across five decades.
    w = np.logspace(-2, 3, 20000)
    w, mag, phase = signal.bode((num, den), w)
    # scipy.signal.bode already returns magnitude in decibels.
    magnitude_db = mag

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.semilogx(w, magnitude_db)
    plt.title('Transfer Function Bode Plot')
    plt.ylabel('$A$ [dB]')
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel(r'$\omega$ [rad/s]')
    plt.ylabel(r'$\phi$ [°]')
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    # Prevent labels from overlapping between the two subplots.
    plt.tight_layout()
    plt.show()


def step_response(num, den):
    """Plot the continuous-time response to a unit-step input."""
    # A fixed time window makes plots comparable across entered systems.
    t = np.linspace(0, 100, 5000)
    system = signal.TransferFunction(num, den)
    t, y = signal.step(system, T=t)

    plt.figure(figsize=(10, 8))
    plt.plot(t, y)
    plt.title('Step Response')
    plt.xlabel(r'$Time [s]$')
    plt.ylabel(r'$Response$')
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()


def impulse_response(num, den):
    """Plot the continuous-time response to a unit-impulse input."""
    # A fixed time window makes plots comparable across entered systems.
    t = np.linspace(0, 100, 5000)
    system = signal.TransferFunction(num, den)
    t, y = signal.impulse(system, T=t)

    plt.figure(figsize=(10, 8))
    plt.plot(t, y)
    plt.title('Impulse Response')
    plt.xlabel(r'$Time [s]$')
    plt.ylabel(r'$Response$')
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()


def poles_zeros_gain(num, den):
    """Print and plot the transfer function's zeros, poles, and gain."""
    # Convert polynomial form into the zero-pole-gain representation.
    zz, pp, kk = signal.tf2zpk(num, den)
    print(f'Zeros = {zz}\nPoles = {pp}\nGain = {kk}')

    plt.figure(figsize=(8, 8))

    # Standard convention: circles represent zeros.
    plt.scatter(
        np.real(zz),
        np.imag(zz),
        marker='o',
        s=100,
        label='Zeros'
    )

    # Standard convention: crosses represent poles.
    plt.scatter(
        np.real(pp),
        np.imag(pp),
        marker='x',
        s=100,
        label='Poles'
    )

    # Draw the real and imaginary axes through the origin.
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)

    plt.xlabel(r"Real Axis ($\sigma$) [$s^{-1}$]")
    plt.ylabel(r"Imaginary Axis ($j\omega$) [$s^{-1}$]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Equal scaling preserves pole/zero geometry in the complex plane.
    plt.axis("equal")

    plt.show()


def root_locus(num, den):
    """Plot closed-loop pole locations as scalar loop gain varies."""
    system = ctrl.TransferFunction(num, den)

    plt.figure(figsize=(10, 8))
    ctrl.root_locus_plot(system)
    plt.grid(True)
    plt.show()


def main():
    """Run the interactive analysis loop until the user chooses to exit."""
    num = None
    den = None

    while True:

        # Request coefficients initially and whenever the user chooses a new
        # transfer function after completing an analysis.
        if num is None or den is None:
            num, den = vnos_koeficientov()

        if all(x == 0 for x in den):
            print("Denominator cannot be zero.")

        option = menu()

        # This educational tool excludes direct-feedthrough terms from time
        # responses, so the numerator order must be below the denominator.
        if option in ("2", "3") and len(num) >= len(den):
            print(
                "Time responses require a strictly proper "
                "transfer function in this tool."
            )
            continue

        if option == "1":
            bode_diagram(num, den)

        elif option == "2":
            step_response(num, den)

        elif option == "3":
            impulse_response(num, den)

        elif option == "4":
            poles_zeros_gain(num, den)

        elif option == "5":
            root_locus(num, den)

        elif option == "6":
            break

        else:
            print("Invalid choice.\n")
            continue

        answer = input(
            "\nDo you want to continue? (y = yes / n = no): "
        ).strip().lower()

        if answer != "y":
            print("Program completed.")
            break

        reuse_answer = input(
            "Do you want to use the same transfer function? "
            "(y = yes / n = no): "
        ).strip().lower()

        if reuse_answer != "y":
            num = None
            den = None


print("Program completed.")


if __name__ == "__main__":
    main()
