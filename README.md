# Educational Machine Learning Website

## Motivation

The goal of this project is to build an **educational website** that allows users to:

- Explore different machine learning algorithms.
- Gain beginner-friendly insights without learning programming.
- Learn machine learning concepts in a **fun and engaging way**.

## Problem Statement

Many beginners struggle with machine learning due to coding requirements and complex concepts. This platform allows users to **experiment and explore ML algorithms interactively**, without needing prior programming knowledge.

## Functional Requirements

- Explore existing machine learning algorithms.
- Add new algorithms with minimal setup.
- Upload Pickle files for models.
- Upload CSV files for datasets.
- Train and test models.
- Receive notifications for updates.
- Subscribe/unsubscribe for algorithm updates.

## Admin's Use-Case Diagram

![Admin Use-Case](https://github.com/waliamehak/LearnML/blob/main/Diagrams/Use%20Cases/Use%20Case%20(Admin).jpg)  

## Learner's Use-Case Diagram

![Learner Use-Case](https://github.com/waliamehak/LearnML/blob/main/Diagrams/Use%20Cases/Use%20Case%20(Learner).jpg)

## Architecture

- Follows **MVC (Model-View-Controller)** architecture.
- Uses **Simple Factory Pattern** for adding new algorithms.
- Uses **Observer Pattern** for notifications to learners.

## Technology Stack

- **PyCharm**: Development IDE for easy project management.
- **Django**: Backend framework implementing MVC architecture.
- **Scikit-learn**: For creating, training, and testing ML algorithms.
- **MongoDB Atlas**: Stores unstructured data as JSON objects.
- **Docker**: Containerized environment for easy setup and consistent development.

## Project Status and Notes

- The original project relied on a MongoDB **M0 cluster** with a snapshot.  
- Recovery steps were attempted: restoring to an M10 cluster, exporting via `mongodump`/`mongorestore`.  
- **Issue**: The snapshot used an **old MongoDB version** with `.wt` files.  
  - Modern MongoDB clusters no longer support this version.  
  - Archived versions for download are not available.  
  - `.wt` files cannot be directly converted to JSON.  
- **Result**: Original dataset is lost. The project **cannot be fully functional** as originally intended.  
- The application structure, code, and functionality are preserved and can be adapted to new datasets or databases.
- **TODO**: Reach out to MongoDB priority team, and seek help. If not possible, manually recreate the whole dataset.

