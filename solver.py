#################################
#								#
#	ACID-BASE REACTION SOLVER 	#
# 								#
#################################

import numpy as np 
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import warnings


class AcidBase:


	def __init__(self, RDB, Conserve_mass):

		'''
		Store the user data locally
		'''
		self.RDB = RDB
		self.Conserve_mass = Conserve_mass
		self.trim_equations()
		self.compute_unknowns()
		self.check_equations()
		self.compute_mass_terms()
		self.temperature = 298
		self.pH_values = []

	def trim_equations(self):
		'''
		Delete the unwanted keys from the equation dictionaries
		'''
		# Remove 'False' reactions 
		[self.RDB.pop(reac) for reac in list(self.RDB) if not self.RDB[reac]['En']]
		# Remove 'False' conservation keys 
		[self.Conserve_mass.pop(specie) for specie in list(self.Conserve_mass) if not self.Conserve_mass[specie]['Enable']]

		return self.RDB, self.Conserve_mass


	def compute_unknowns(self):
		'''
		Figure out what the unknows are
		'''

		# Add keys for the coefficients and the variables
		for reaction in self.RDB:
			self.RDB[reaction]['coeff'] = []
			self.RDB[reaction]['vars']  = []

		# Add keys for the coefficients and the variables
		for mass in self.Conserve_mass:
			self.Conserve_mass[mass]['species'] = []
			self.Conserve_mass[mass]['coeff'] = []


		# Figure out the variables in the system
		# Create dictionary to store the data
		self.unknowns = {}

		# Loop through all the reactions
		for reaction in self.RDB:

				# Split the reaction into their n terms
				temp = self.RDB[reaction]['eq'].replace('<=>', ' ').replace('&', ' ').split()

				# Loop through the n terms of each reaction
				for index,spec in enumerate(temp):

					# Break each term into stoichiometric coefficients, variable name and charge of molecule/ion
					stoic, var, char = spec.split('_')

					if self.RDB[reaction]['Tp'] != 'AD' and index==0:
						# print('Bad apple... ', var, '... Do not add!')
						pass

					elif var == 'H2O':
						# print('Bad apple... ', var, '... Do not add!')
						pass

					else:
						# Add data to the reactions' database dictionary
						self.RDB[reaction]['coeff'].append( int(stoic) )
						self.RDB[reaction]['vars'].append( var+'_'+char )

						# Check if the variable already exists
						if var in self.unknowns:
							pass

						# If variable doesn't exist , then add the variable
						else:
							self.unknowns[var+'_'+char] = {}
							self.unknowns[var+'_'+char]['value'] = 0.
							self.unknowns[var+'_'+char]['charge'] = int(char)

		# Create variables for the solutions
		for unknown in self.unknowns:
			self.unknowns[unknown]['solution'] = []

		return self.RDB, self.Conserve_mass, self.unknowns


	def check_equations(self):
		'''
		Check if the system is balanced: unknowns vs equations
		'''

		# Calculate the number of equilibrium equations
		No_equilibrium = len(self.RDB)

		# Calculate the number of mass equations chosen
		No_mass = len(self.Conserve_mass)

		# Calculate the number of unknows
		No_unknowns  = len(self.unknowns)

		# Total number of equations, including charge
		Total = No_mass + No_equilibrium + 1
		print('You have {} unknowns and {} equations: {} for equilibrium, {} for mass and 1 for charge.'.format(No_unknowns, Total, No_equilibrium, No_mass))

		# Check whether the system is over- or under-determinate
		if No_unknowns > Total:
			choice = input('Your system is under-determinate. Do you wish to continue? Y/n \n')
			if choice == 'n':
				quit()

		elif No_unknowns < Total:
			choice = input('Your system is over-determinate. Your system is over-determinate. Do you wish to continue? Y/n \n')
			if choice == 'n':
				quit()

		else:
			print('The system is determinate.')

		self.No_equilibrium = No_equilibrium
		self.No_mass = No_mass
		self.No_unknowns = No_unknowns


	def compute_mass_terms(self):
		'''
		Figure out which variables are needed for mass conservation
		'''
		# Figure out the items that must be conserved for mass, and their coefficients
		# Loop through the item that can be conserved
		for mass in self.Conserve_mass:

			# Check if they're enabled
			if self.Conserve_mass[mass]['Enable']:

				# For the enabled ones, let's find what unkowns they're present in
				for item in self.unknowns:
					position = item.find(mass)

					# If we find a successful unknown, let's append it to the list!
					if position >= 0:
						self.Conserve_mass[mass]['species'].append(item)
						parenthesis = position+len(mass)

						# Let's figure out their coefficient
						if item[parenthesis] == ')':
							self.Conserve_mass[mass]['coeff'].append(int(item[parenthesis+1])) 
						else:
							self.Conserve_mass[mass]['coeff'].append(1.)

		return self.Conserve_mass


	def functions(self, solutions, *args):
		'''
		This is the function required by the scipy solver
		'''

		# Assign the solver solution to my working variables
		for index, unknown in enumerate(self.unknowns):
			self.unknowns[unknown]['value'] = solutions[index]

		# Create an array to hold my functions
		f = np.zeros(self.No_unknowns)

		# Loop over the reactions
		for index, reaction in enumerate(self.RDB):

			# Loop over the molecules and coefficients in each reaction
			f[index] = self.RDB[reaction]['pK'] + np.sum([ coeff*self.unknowns[molecule]['value'] for molecule, coeff in zip(self.RDB[reaction]['vars'], self.RDB[reaction]['coeff']) ])

		# Conserve mass
		k = -2
		for specie in self.Conserve_mass:
			values = np.asarray([ self.unknowns[spec]['value'] for spec in self.Conserve_mass[specie]['species'] ])
			coeffs = np.asarray( self.Conserve_mass[specie]['coeff'] )
			f[k] = np.sum(coeffs*10.**values) - self.Conserve_mass[specie]['C_init']
			k -= 1

		# Conserve charge
		f[-1] = np.sum([ self.unknowns[unknown]['charge']*10.**self.unknowns[unknown]['value'] for unknown in self.unknowns ])

		return f


	def solve_system(self, guess, *args):
		'''
		Solves the system using python's scipy library, fsolve module
		'''

		warnings.filterwarnings("default", category=RuntimeWarning) 

		z = fsolve(self.functions, guess)

		'''
			!!!   VERY IMPORTANT NOTE   !!!

		If you choose to use the full output here, then you must
		change the sweeping initial guess to:
			>>> init_guess = mySolution[0]
		instead of just:
			>>> init_guess = mySolution
		Otherwise, it will give an error!
		'''
		return z


	def solve_system_single(self):
		'''
		Solves the system using python's scipy library, fsolve module
		'''
		warnings.filterwarnings("ignore", category=RuntimeWarning) 

		factor = -10. 												# Starting factor for the initial guess
		converged = 0 												# Equals 1 if convergence happens

		while converged != 1:
			init_guess = factor*np.ones(self.No_unknowns) 		# The initial guess, based on the "factor"
			z = fsolve(self.functions, init_guess, full_output=1 )
			converged = z[-2] 										# Output from solver, which indicates convergence success (or not
			factor += 0.5 											# Increment in factor

			if converged == 1:
				print('The solution converged for the factor: ', factor)
				return z
				break

			if factor > 20.:						# Leave the loop if no convergence until factor 5
				print('Did not converge... :(')
				return z
				break




if __name__ == '__main__':

	Conserve_mass = {
		'PO4': {'Enable':1, 'C_init': 1.}}

	RDB = {
		# Water dissociation
		'a00': {'eq': '-1_H2O_0 <=> 1_H_+1 & 1_OH_-1', 'En': 1, 'pK':14.00, 'Tp':'WD'},

		# Solutions of orthophosphoric acid
		'b00': {'eq': '-1_H3PO4_0   <=> 1_H_+1  & 1_H2PO4_-1',	'En': 1, 'pK':02.15, 'Tp':'AD'},
		'b01': {'eq': '-1_H2PO4_-1  <=> 1_H_+1  & 1_HPO4_-2',	'En': 1, 'pK':07.20, 'Tp':'AD'},
		'b02': {'eq': '-1_HPO4_-2   <=> 1_H_+1  & 1_PO4_-3',	'En': 1, 'pK':12.35, 'Tp':'AD'}} 

	system = AcidBase(RDB, Conserve_mass) 					# Create the system of equations

	# Example of successfully solving a multiple case with errors
	init_guess = -2.*np.ones(system.No_unknowns)
	mySolution = system.solve_system(init_guess)
	pH = -system.unknowns['H_+1']['value']
	print('The pH is: {0:8.4f}'.format(pH) )

	# Example of unsuccessfully solving a multiple case with errors
	init_guess = -20.*np.ones(system.No_unknowns)
	mySolution = system.solve_system(init_guess)
	pH = -system.unknowns['H_+1']['value']
	print('The pH is: {0:8.4f}'.format(pH) )


	# Example of successfully solving a single case with errors
	system.solve_system_single()
	pH = -system.unknowns['H_+1']['value']
	print('The pH is: {0:8.4f}'.format(pH) )

	
