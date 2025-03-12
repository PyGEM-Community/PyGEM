.. pygem documentation master file, created by
   sphinx-quickstart on Sat Jun 10 23:30:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyGEM's documentation!
=================================
The Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that is designed to model the transient evolution of glaciers on regional 
and global scales. Each glacier is modeled independently using a given time step and elevation bins. The model computes the climatic mass balance (i.e., snow accumulation minus 
melt plus refreezing) for each elevation bin and each monthly time step. Glacier geometry is updated annually. The model outputs a variety of data including monthly mass balance 
and its components (accumulation, melt, refreezing, frontal ablation, glacier runoff), and annual volume, volume below sea level, and area.

PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., ablation, accumulation, refreezing, glacier dynamics).
The most recent version of PyGEM, published in *Science* `(Rounce et al., 2023) <https://www.science.org/doi/10.1126/science.abo1324>`_, has been made compatible with the 
Open Global Glacier Model `(OGGM) <https://oggm.org/>`_ to both leverage the pre-processing tools (e.g., digital elevation models, glacier characteristics) and their advances 
with respect to modeling glacier dynamics and ice thickness inversions.

.. admonition:: Note for new users:

   - Looking for a quick overview? Check out the :doc:`model structure and workflow <model_structure>`.
   - Want to read some studies? Check out our :doc:`publications <publications>`.
   - Want to see what PyGEM can do? Check out this presentation about PyGEM's latest developments:

   .. raw:: html

      <iframe width="672" height="378" src="https://www.youtube.com/embed/gaGzEIjIJlc?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


.. toctree::
   :maxdepth: 1
   :caption: Overview:

   introduction
   model_structure
   model_inputs
   mb_parameterizations
   dynamics_parameterizations
   runoff
   calibration_options
   bias_corrections
   initial_conditions
   limitations
   publications
   faqs

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   install_pygem
   test_pygem
   scripts_overview
   model_output

.. toctree::
   :maxdepth: 1
   :caption: Contributing:

   contributing
   citing

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
